"""Multi-device and multi-host training: meshes, shardings, global batches.

The model is JAX's multi-controller SPMD: one process per host, every process
runs the same program, and arrays are *global* -- each process holds only the
shards on its own devices. `jit` with `NamedSharding`s does the rest: XLA
inserts the gradient all-reduce and the metric reductions, so a task's
`train_step` -- written as if for one device, with means over the whole batch --
is already a data-parallel step. No task code mentions devices.

What does need care lives here and in `core.loop`:

* **Batches.** Each process loads only its slice of every global batch, and
  `Partitioner.to_global` stitches the slices into one sharded array.
* **Eval.** Every eval batch is a collective, so every process must run the
  same number of them at the same shape, or the run hangs. `eval_batches` pads
  short batches with masked-out rows and runs empty, fully masked ones on
  processes that ran out early. The mask reaches `Task.eval_step`, so metrics
  still count every real example exactly once.
* **Control.** A loop that stops on one host and not another deadlocks.
  `host_any`/`host_all` turn per-host decisions into global ones.
* **Side effects.** Files are written by the primary process only
  (`is_primary`); checkpoints are written by all, since orbax coordinates them.
"""

import enum
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, cast

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from attrs import validators as v
from flax import struct
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from core.types import HostOrDeviceArray, PRNGKey

# --- Launch ------------------------------------------------------------------


class CpuCollectives(enum.Enum):
    """Cross-process collectives on CPU. Needed only to fake multi-host on CPU."""

    GLOO = "gloo"
    MPI = "mpi"


@attrs.frozen(kw_only=True)
class SingleProcess:
    """One process driving all local devices. Nothing to initialize."""


@attrs.frozen(kw_only=True)
class ClusterAuto:
    """Let JAX read the cluster from the environment (SLURM, GKE, Cloud TPU, MPI)."""


@attrs.frozen(kw_only=True)
class Coordinator:
    """Explicit rendezvous: every process is given the same address and count."""

    address: str
    num_processes: int = attrs.field(validator=v.gt(0))
    process_id: int = attrs.field(validator=v.ge(0))
    cpu_collectives: CpuCollectives | None = None

    @process_id.validator  # pyright: ignore[reportAttributeAccessIssue, reportUntypedFunctionDecorator]
    def _in_range(self, attribute: "attrs.Attribute[int]", value: int) -> None:
        if value >= self.num_processes:
            raise ValueError(
                f"process_id={value} outside [0, num_processes={self.num_processes})"
            )


type Launch = SingleProcess | ClusterAuto | Coordinator


def initialize(launch: Launch) -> None:
    """Join the job. Call once, first thing in `main`, before touching devices."""
    match launch:
        case SingleProcess():
            return
        case ClusterAuto():
            jax.distributed.initialize()
        case Coordinator():
            if launch.cpu_collectives is not None:
                jax.config.update(
                    "jax_cpu_collectives_implementation", launch.cpu_collectives.value
                )
            jax.distributed.initialize(
                coordinator_address=launch.address,
                num_processes=launch.num_processes,
                process_id=launch.process_id,
            )


@attrs.frozen(kw_only=True)
class Shard:
    """Which slice of a dataset this process reads: rows `index::count`."""

    index: int = attrs.field(validator=v.ge(0))
    count: int = attrs.field(validator=v.gt(0))

    @classmethod
    def current(cls) -> "Shard":
        """One slice per process -- what every pipeline defaults to."""
        return cls(index=jax.process_index(), count=jax.process_count())


def is_primary() -> bool:
    """The process that writes logs, configs, and other single-copy files."""
    return jax.process_index() == 0


def host_any(value: bool) -> bool:
    """True on every process if `value` is True on any. Free when single-process."""
    if jax.process_count() == 1:
        return value
    gathered = multihost_utils.process_allgather(np.asarray(value))
    return bool(np.any(gathered))


def host_all(value: bool) -> bool:
    return not host_any(not value)


def host_max(value: int) -> int:
    if jax.process_count() == 1:
        return value
    return int(np.max(multihost_utils.process_allgather(np.asarray(value))))


# --- Mesh --------------------------------------------------------------------


class Axis(enum.Enum):
    """Mesh axes. Values are the names JAX sees; nothing else spells them.

    DATA and FSDP both split the batch. FSDP additionally splits parameters and
    optimizer state (`Fsdp`); MODEL is for tensor parallelism via `Custom` specs.
    """

    DATA = "data"
    FSDP = "fsdp"
    MODEL = "model"


# The batch axis is split over every data-parallel mesh axis.
BATCH_AXES: tuple[str, ...] = (Axis.DATA.value, Axis.FSDP.value)


def axis_spec(*axes: Axis | tuple[Axis, ...] | None) -> P:
    """A `PartitionSpec` from `Axis` members -- `axis_spec(None, Axis.MODEL)`."""

    def name(a: Axis | tuple[Axis, ...] | None) -> str | tuple[str, ...] | None:
        match a:
            case None:
                return None
            case Axis():
                return a.value
            case tuple():
                return tuple(x.value for x in a)

    return P(*(name(a) for a in axes))


@attrs.frozen(kw_only=True)
class MeshConfig:
    """Mesh shape. Exactly one axis may be -1, meaning "every remaining device"."""

    data: int = attrs.field(default=-1, validator=v.or_(v.gt(0), v.in_((-1,))))
    fsdp: int = attrs.field(default=1, validator=v.or_(v.gt(0), v.in_((-1,))))
    model: int = attrs.field(default=1, validator=v.or_(v.gt(0), v.in_((-1,))))

    def __attrs_post_init__(self) -> None:
        if (self.data, self.fsdp, self.model).count(-1) > 1:
            raise ValueError(f"at most one mesh axis may be -1: {self}")

    def shape(self, num_devices: int) -> tuple[int, int, int]:
        sizes = (self.data, self.fsdp, self.model)
        known = math.prod(s for s in sizes if s != -1)
        if num_devices % known:
            raise ValueError(f"{num_devices} devices do not divide into {self}")
        filled = tuple(num_devices // known if s == -1 else s for s in sizes)
        if math.prod(filled) != num_devices:
            raise ValueError(
                f"mesh {filled} uses {math.prod(filled)} devices, have {num_devices}"
            )
        return cast(tuple[int, int, int], filled)


# --- State sharding strategies -------------------------------------------------


@attrs.frozen(kw_only=True)
class Replicated:
    """Every device holds the whole state: plain data parallelism."""


@attrs.frozen(kw_only=True)
class Fsdp:
    """Shard each large array's largest evenly divisible dim over `Axis.FSDP`.

    Shape-driven, so optimizer moments follow their parameters with no extra
    rules. Arrays below `min_size` elements stay replicated: splitting a bias
    costs more in communication than it saves in memory.
    """

    min_size: int = attrs.field(default=2**14, validator=v.gt(0))


@attrs.frozen(kw_only=True)
class Custom[StateT]:
    """Hand-written specs: abstract state -> a same-structure tree of `PartitionSpec`.

    For layouts a heuristic cannot guess: tensor parallelism over `Axis.MODEL`,
    or an RL task sharding its environments over the batch axes.
    """

    specs: Callable[[StateT], Any]


@attrs.frozen(kw_only=True)
class Rule:
    """Map one logical axis -- a model's own enum member -- to a mesh axis.

    `mesh=None` keeps that logical axis unsplit (the default for any logical
    axis without a rule).
    """

    logical: enum.Enum
    mesh: Axis | None


def _unique_logical(
    instance: "Logical",
    attribute: "attrs.Attribute[tuple[Rule, ...]]",
    value: tuple[Rule, ...],
) -> None:
    names = [r.logical for r in value]
    if len(set(names)) != len(names):
        raise ValueError(f"a logical axis has more than one rule: {names}")


@attrs.frozen(kw_only=True)
class Logical:
    """Layout from the model's own annotations, mapped through `rules`.

    Layers declare what each weight dim *means* -- `partitioned(init,
    Feature.EMBED, Feature.HIDDEN)` -- and the rules decide where each meaning
    lives on the mesh. The same model then trains replicated, FSDP, tensor
    parallel or both by swapping rules, with no change to its code.

    Optimizer moments are covered for free: optax maps over the annotated
    parameter boxes, so `mu`/`nu` carry the same annotations. Unannotated
    leaves (step, rng, LayerNorm scales, ...) are replicated.
    """

    rules: tuple[Rule, ...] = attrs.field(default=(), validator=_unique_logical)

    def mesh_axis(self, logical: str | None) -> str | None:
        for rule in self.rules:
            if rule.logical.value == logical:
                return None if rule.mesh is None else rule.mesh.value
        return None


type StateSharding[StateT] = Replicated | Fsdp | Custom[StateT] | Logical


def partitioned[**Ps](
    init: Callable[Ps, jax.Array], *axes: enum.Enum | None
) -> Callable[Ps, jax.Array]:
    """Annotate a Linen initializer with one logical axis (or None) per dim."""
    names = tuple(None if a is None else a.value for a in axes)
    return cast(Callable[Ps, jax.Array], nn.with_logical_partitioning(init, names))


def _is_box(x: object) -> bool:
    return isinstance(x, nn.LogicallyPartitioned)


def logical_specs(abstract: object, strategy: Logical) -> Any:
    """Specs for `Logical`: annotated leaves through the rules, the rest replicated."""

    def spec(x: object) -> object:
        if isinstance(x, nn.LogicallyPartitioned):
            names = x.names
            axes = [strategy.mesh_axis(n) for n in names]
            used = [a for a in axes if a is not None]
            if len(set(used)) != len(used):
                # Logical axes may share a mesh axis, but not on one array.
                raise ValueError(f"{names} put two dims of one array on {used}")
            return x.replace_boxed(P(*axes))
        return P()

    return jax.tree.map(spec, abstract, is_leaf=_is_box)


def fsdp_spec(shape: Sequence[int], axis_size: int, min_size: int) -> P:
    """The `Fsdp` rule for one array."""
    if axis_size == 1 or math.prod(shape) < min_size:
        return P()
    candidates = [i for i, d in enumerate(shape) if d % axis_size == 0]
    if not candidates:
        return P()
    dim = max(candidates, key=lambda i: shape[i])
    return P(*(Axis.FSDP.value if i == dim else None for i in range(len(shape))))


def _is_spec(x: object) -> bool:
    return isinstance(x, P)


# --- Batches -----------------------------------------------------------------


def batch_size_of(batch: object) -> int | None:
    """Leading dim shared by every leaf; None for a leafless batch (`NoBatch`)."""
    sizes = {np.shape(leaf)[0] for leaf in jax.tree.leaves(batch)}
    if not sizes:
        return None
    if len(sizes) > 1:
        raise ValueError(f"batch leaves disagree on the batch dim: {sorted(sizes)}")
    return sizes.pop()


def pad_batch[BatchT](batch: BatchT, size: int) -> BatchT:
    """Pad every leaf's leading dim to `size` with zeros, on host."""

    def pad(x: HostOrDeviceArray) -> np.ndarray:
        x = np.asarray(x)
        widths = [(0, size - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
        return np.pad(x, widths)

    return jax.tree.map(pad, batch)


@struct.dataclass
class EvalBatch[BatchT]:
    """A global eval batch and which of its rows are real examples."""

    batch: BatchT
    mask: jax.Array


# --- Partitioner -------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Partitioner:
    """A mesh plus everything that places arrays on it."""

    mesh: Mesh

    @classmethod
    def create(
        cls,
        config: MeshConfig = MeshConfig(),
        # `jax.Device` is a runtime alias the type checker cannot use as a type.
        devices: Sequence[Any] | None = None,
    ) -> "Partitioner":
        """Over every device in the job, unless `devices` says otherwise."""
        devices = jax.devices() if devices is None else devices
        mesh = jax.make_mesh(
            config.shape(len(devices)),
            tuple(a.value for a in Axis),
            # Auto: GSPMD propagates shardings through the step, which is what
            # lets task code stay sharding-unaware.
            axis_types=(AxisType.Auto,) * len(Axis),
            devices=devices,
        )
        return cls(mesh=mesh)

    def size(self, axis: Axis) -> int:
        return self.mesh.shape[axis.value]

    @property
    def data_shards(self) -> int:
        """How many ways the batch is split."""
        return self.size(Axis.DATA) * self.size(Axis.FSDP)

    @property
    def local_batch_multiple(self) -> int:
        """A process-local batch must be a multiple of this."""
        return max(1, self.data_shards // jax.process_count())

    @property
    def replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    def batch_sharding(self, leading: int = 0) -> NamedSharding:
        """Batch dim split over the data axes, after `leading` unsplit dims.

        `leading=1` for the `(steps_per_call, batch, ...)` stacks the loop feeds.
        """
        return NamedSharding(self.mesh, P(*([None] * leading), BATCH_AXES))

    def to_global[BatchT](self, local: BatchT, leading: int = 0) -> BatchT:
        """This process's slice of a batch -> the global, sharded batch.

        Every process passes its own rows; the global batch is their
        concatenation in process order.
        """
        sharding = self.batch_sharding(leading)

        def place(x: HostOrDeviceArray) -> jax.Array:
            x = np.asarray(x)
            if x.shape[leading] % self.local_batch_multiple:
                raise ValueError(
                    f"process-local batch of {x.shape[leading]} does not split over "
                    f"{self.local_batch_multiple} local data shards"
                )
            return jax.make_array_from_process_local_data(sharding, x)

        return jax.tree.map(place, local)

    def replicate[T](self, tree: T) -> T:
        """Host values every process agrees on -> replicated global arrays."""
        return jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(
                self.replicated, np.asarray(x)
            ),
            tree,
        )

    def specs[StateT](self, abstract: StateT, strategy: StateSharding[StateT]) -> Any:
        """A tree of `PartitionSpec` shaped like `abstract`."""
        match strategy:
            case Replicated():
                return jax.tree.map(lambda _: P(), abstract)
            case Fsdp(min_size=min_size):
                axis = self.size(Axis.FSDP)
                return jax.tree.map(
                    lambda x: fsdp_spec(np.shape(x), axis, min_size), abstract
                )
            case Custom(specs=fn):
                return fn(abstract)
            case Logical():
                return logical_specs(abstract, strategy)

    def shardings[StateT](
        self, abstract: StateT, strategy: StateSharding[StateT]
    ) -> Any:
        """A tree of `NamedSharding` shaped like `abstract`."""
        return jax.tree.map(
            lambda p: NamedSharding(self.mesh, p),
            self.specs(abstract, strategy),
            is_leaf=_is_spec,
        )

    def abstract_state[StateT, BatchT](
        self,
        init: Callable[[PRNGKey, BatchT], StateT],
        rng: PRNGKey,
        example: BatchT,
        strategy: StateSharding[StateT],
    ) -> StateT:
        """`jax.eval_shape(init)` with each leaf carrying its sharding.

        What orbax needs to restore straight into the sharded layout, rather
        than gathering a full copy onto one device first.
        """
        abstract = jax.eval_shape(init, rng, example)
        shardings = self.shardings(abstract, strategy)
        return jax.tree.map(
            lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
            abstract,
            shardings,
        )

    def eval_batches[BatchT](
        self, batches: Iterable[BatchT]
    ) -> Iterator[EvalBatch[BatchT]]:
        """Process-local eval batches -> equal-shaped, masked global batches.

        All processes agree on a batch size (the largest first batch, rounded up
        to split over the local data shards) and keep going until every one of
        them has run out, feeding fully masked batches once their own data ends.
        """
        it = iter(batches)
        first = next(it, None)
        has_any = first is not None
        if not host_all(has_any):
            raise ValueError("every process needs at least one eval batch")
        assert first is not None
        local = batch_size_of(first)
        if local is None:  # leafless: nothing to pad, nothing to split
            yield from self._leafless(first, it)
            return

        multiple = self.local_batch_multiple
        size = math.ceil(host_max(local) / multiple) * multiple
        template = pad_batch(first, size)
        current: BatchT | None = first
        while host_any(current is not None):
            if current is None:
                batch, real = template, 0
            else:
                real = batch_size_of(current) or 0
                if real > size:
                    raise ValueError(
                        f"eval batch of {real} exceeds the agreed size {size}; "
                        "the first eval batch must be the largest"
                    )
                batch = pad_batch(current, size)
            mask = np.arange(size) < real
            yield EvalBatch(
                batch=self.to_global(batch),
                mask=cast(jax.Array, self.to_global(mask)),
            )
            current = next(it, None)

    def _leafless[BatchT](
        self, first: BatchT, rest: Iterator[BatchT]
    ) -> Iterator[EvalBatch[BatchT]]:
        empty = cast(jax.Array, self.replicate(np.zeros((0,), dtype=bool)))
        current: BatchT | None = first
        while host_any(current is not None):
            yield EvalBatch(batch=first if current is None else current, mask=empty)
            current = next(rest, None)


def all_true(batch: object) -> jax.Array:
    """The mask for a batch with no padding; `(b,)`, or `(0,)` if leafless."""
    return jnp.ones((batch_size_of(batch) or 0,), dtype=bool)
