"""Flags every entry point shares for joining a multi-host job.

    # one host, all local GPUs: nothing to pass
    python -m tasks.mnist.main --workdir=...
    # SLURM / GKE / Cloud TPU: JAX reads the cluster from the environment
    python -m tasks.mnist.main --workdir=... --cluster_auto
    # anything else: every process gets the same address and count
    python -m tasks.mnist.main --workdir=... \\
        --coordinator=host0:1234 --num_processes=4 --process_id=$RANK
    # fake two hosts on this machine (run both)
    XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m tasks.mnist.main \
        --workdir=... --coordinator=localhost:1234 --num_processes=2 \
        --process_id=0 --cpu_collectives=gloo
"""

from absl import flags

from commmon.checks import set_shape_checks
from core.distributed import (
    ClusterAuto,
    Coordinator,
    CpuCollectives,
    Launch,
    SingleProcess,
)

_CLUSTER_AUTO = flags.DEFINE_bool(
    "cluster_auto", False, "Join a cluster JAX can detect (SLURM, GKE, TPU)."
)
_COORDINATOR = flags.DEFINE_string(
    "coordinator", None, "host:port of process 0, for an explicit multi-host job."
)
_NUM_PROCESSES = flags.DEFINE_integer("num_processes", None, "Processes in the job.")
_PROCESS_ID = flags.DEFINE_integer("process_id", None, "This process's index.")
_CPU_COLLECTIVES = flags.DEFINE_enum_class(
    "cpu_collectives",
    None,
    CpuCollectives,
    "Cross-process collectives on CPU; only for faking multi-host on one machine.",
)


_SHAPE_CHECKS = flags.DEFINE_bool(
    "shape_checks",
    False,
    "Run eincheck's runtime shape checks. Off by default for throughput: ingest "
    "already validated every record, and the per-example re-checks were ~80% of "
    "the input pipeline's CPU. Turn on when changing schemas or transforms.",
)


def shape_checks_from_flags() -> bool:
    """Apply --shape_checks to this process; returns it, for the data pipeline
    to apply in its worker processes too (see `data.transforms.ShapeChecks`)."""
    set_shape_checks(_SHAPE_CHECKS.value)
    return _SHAPE_CHECKS.value


def from_flags() -> Launch:
    if _COORDINATOR.value is not None:
        if _NUM_PROCESSES.value is None or _PROCESS_ID.value is None:
            raise flags.ValidationError(
                "--coordinator needs --num_processes and --process_id"
            )
        return Coordinator(
            address=_COORDINATOR.value,
            num_processes=_NUM_PROCESSES.value,
            process_id=_PROCESS_ID.value,
            cpu_collectives=_CPU_COLLECTIVES.value,
        )
    if _CLUSTER_AUTO.value:
        return ClusterAuto()
    return SingleProcess()
