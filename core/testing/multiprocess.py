"""Worker: one of N processes of a fake multi-host job on localhost.

    python -m core.testing.multiprocess <process_id> <num_processes> <port> \\
        <workdir> <out.npz>

Each process fakes `LOCAL_DEVICES` CPU devices and joins the job over gloo, so
the run exercises what real multi-host does: global arrays built from
per-process slices, cross-process collectives in every step, orbax writing one
checkpoint from all processes, and host-side decisions kept in agreement.

Data is deliberately uneven at eval -- process 0 holds three eval batches,
process 1 two, the last of them short -- which deadlocks unless eval padding
and lockstep work.
"""

import os
import pathlib
import socket
import subprocess
import sys

import attrs
import jax
import numpy as np
import orbax.checkpoint as ocp

from core.callbacks import (
    AtEnd,
    Callback,
    Checkpoint,
    Control,
    Evaluate,
    Every,
    LogMetrics,
    StepEvent,
    evaluate,
    restore_latest,
)
from core.config import ExperimentConfig, LoopConfig
from core.distributed import (
    Coordinator,
    CpuCollectives,
    MeshConfig,
    Partitioner,
    Replicated,
    initialize,
)
from core.export import run_writer
from core.loop import fit
from core.optim import AdamW, Constant, OptimConfig
from core.testing import toy

LOCAL_DEVICES = 2
STOP_AT = 10


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def launch(
    num_processes: int, workdir: pathlib.Path, timeout: float = 600
) -> list[str]:
    """Run the job; returns each process's `.npz` path, in process order."""
    port = free_port()
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={LOCAL_DEVICES}",
        "JAX_PLATFORMS": "cpu",
    }
    outs = [str(workdir / f"process_{i}.npz") for i in range(num_processes)]
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "core.testing.multiprocess",
                str(i),
                str(num_processes),
                str(port),
                str(workdir),
                outs[i],
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for i in range(num_processes)
    ]
    logs: list[str] = []
    try:
        for p in procs:
            out, _ = p.communicate(timeout=timeout)
            logs.append(out)
    finally:
        for p in procs:
            p.kill()
    for i, p in enumerate(procs):
        if p.returncode:
            raise RuntimeError(f"process {i} failed:\n{logs[i][-4000:]}")
    return outs


def local_eval_rows(index: int, count: int) -> slice:
    """Uneven on purpose: process 0 takes 24 of the 37 eval rows."""
    assert count == 2
    return slice(0, 24) if index == 0 else slice(24, None)


@attrs.frozen(kw_only=True)
class StopOnOneProcess(Callback[toy.ToyState, toy.ToyMetrics, None]):
    """Asks to stop on process 1 only -- the loop must stop everywhere."""

    def begin(self, state: toy.ToyState) -> None:
        return None

    def fire(
        self, cb: None, event: StepEvent[toy.ToyState, toy.ToyMetrics]
    ) -> tuple[None, Control]:
        stop = jax.process_index() == 1 and event.step >= STOP_AT
        return None, Control.STOP if stop else Control.CONTINUE


def main(
    process_id: int, num_processes: int, port: int, workdir: str, out: str
) -> None:
    # Joining the job only has to precede touching devices, not importing jax.
    initialize(
        Coordinator(
            address=f"localhost:{port}",
            num_processes=num_processes,
            process_id=process_id,
            cpu_collectives=CpuCollectives.GLOO,
        )
    )

    index, count = jax.process_index(), jax.process_count()
    assert len(jax.devices()) == LOCAL_DEVICES * count
    partitioner = Partitioner.create(MeshConfig())
    task = toy.ToyTask()
    work = pathlib.Path(workdir)

    def train():
        return (toy.local_slice(b, index, count) for b in toy.global_batches())

    eval_local = toy.eval_batches(
        jax.tree.map(lambda a: a[local_eval_rows(index, count)], toy.eval_set()), 8
    )
    loop = LoopConfig(num_steps=20, steps_per_call=5)
    cfg = ExperimentConfig(
        task=toy.ToyConfig(),
        loop=loop,
        optim=OptimConfig(optimizer=AdamW(lr=Constant(value=1e-2))),
        workdir=work,
    )
    writer = run_writer(cfg)
    manager = ocp.CheckpointManager(work.absolute() / "ckpt")
    state = fit(
        task,
        config=loop,
        train=train(),
        partitioner=partitioner,
        callbacks=(
            LogMetrics(trigger=Every(n=5), writer=writer),
            Evaluate(
                trigger=Every(n=10),
                task=task,
                batches=eval_local,
                writer=writer,
                partitioner=partitioner,
            ),
            Checkpoint(trigger=AtEnd(), manager=manager),
        ),
    )
    result = evaluate(task, state, eval_local, partitioner=partitioner)

    example = partitioner.to_global(next(train()))
    abstract = partitioner.abstract_state(
        task.init, jax.random.key(0), example, Replicated()
    )
    restored = restore_latest(manager, abstract)
    assert restored is not None
    restored_equal = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(restored.params), jax.tree.leaves(state.params), strict=True
        )
    )

    stopped = fit(
        task,
        config=LoopConfig(num_steps=40, steps_per_call=5),
        train=train(),
        partitioner=partitioner,
        callbacks=(StopOnOneProcess(trigger=Every(n=5)),),
    )

    np.savez(
        out,
        np.asarray(index),
        np.asarray(float(result.loss.compute())),
        np.asarray(int(result.loss.count)),
        np.asarray(restored_equal),
        np.asarray(int(task.step_of(stopped))),
        np.asarray(len(writer.writers)),
        *toy.flat_params(state),
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
