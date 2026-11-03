"""Worker: one process faking N CPU devices.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m core.testing.multidevice core.testing.toy:fsdp out.npz

Runs `module:function` -- which takes nothing and returns a tuple of arrays --
and saves them to an `.npz` for the test to compare against an in-process,
single-device reference. The device count has to be fixed before jax starts,
hence a fresh process rather than a fixture.
"""

import importlib
import os
import subprocess
import sys
from collections.abc import Callable

import numpy as np

DEVICES_FLAG = "--xla_force_host_platform_device_count"


def run_with_devices(target: str, out: str, devices: int, timeout: float = 600) -> None:
    """Run `target` in a fresh process with `devices` fake CPU devices."""
    env = {
        **os.environ,
        "XLA_FLAGS": f"{DEVICES_FLAG}={devices}",
        "JAX_PLATFORMS": "cpu",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "core.testing.multidevice", target, out],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode:
        raise RuntimeError(f"{target} failed:\n{proc.stderr[-4000:]}")


def load(out: str) -> tuple[np.ndarray, ...]:
    with np.load(out) as npz:
        return tuple(npz[f"arr_{i}"] for i in range(len(npz.files)))


def main(target: str, out: str) -> None:
    module, name = target.split(":")
    fn: Callable[[], tuple[np.ndarray, ...]] = getattr(
        importlib.import_module(module), name
    )
    np.savez(out, *fn())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
