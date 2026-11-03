"""Typed metric containers built from clu metrics.

A task's metrics are a `Metrics` subclass whose fields are clu `Metric`s:

    @struct.dataclass
    class MnistMetrics(Metrics):
        loss: clu_metrics.Average
        accuracy: clu_metrics.Accuracy

and are built by plain construction -- `MnistMetrics(loss=Average.from_model_output(
loss), ...)` -- never through clu's string-keyed `Collection.single_from_model_output
(**outputs)` or `Average.from_output("loss")`. Code reads a value as
`metrics.loss.compute()`, which the type checker follows.

We do not subclass `clu_metrics.Collection`: its positional `_reduction_counter`
field gets in the way of direct construction, and its `merge`/`compute` work
through name-keyed dicts. The helpers here do the same job by walking the pytree.

Field names become strings in exactly one place, `scalars`, at the logging
boundary.
"""

import dataclasses
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import jax
from clu import metrics as clu_metrics
from flax import struct


@runtime_checkable
class Labeled(Protocol):
    """A metric whose value is several named scalars -- a grid of hit rates
    over radii and horizons, say -- rather than one. `scalars` names each as
    `<field>/<label>`."""

    def named_scalars(self) -> Iterator[tuple[str, float]]: ...


@struct.dataclass
class Metrics:
    """Base of every metrics container. Subclasses add clu `Metric` fields."""


def _is_metric(x: object) -> bool:
    return isinstance(x, clu_metrics.Metric)


def merge[M: Metrics](a: M, b: M) -> M:
    """Field-wise `Metric.merge`. Works under jit and on host."""
    return jax.tree.map(lambda x, y: x.merge(y), a, b, is_leaf=_is_metric)


def reduce_stacked[M: Metrics](m: M) -> M:
    """Merge a stack of metrics along the leading axis, e.g. the output of a scan."""
    return jax.tree.map(lambda x: x.reduce(), m, is_leaf=_is_metric)


def scalars(m: Metrics, prefix: str = "") -> Iterator[tuple[str, float]]:
    """`(field name, value)` for every metric, recursing into nested containers.

    Host side only: forces a device sync. Nested containers are named
    `outer/inner`.
    """
    for field in dataclasses.fields(m):
        value = getattr(m, field.name)
        name = f"{prefix}{field.name}"
        if isinstance(value, Metrics):
            yield from scalars(value, prefix=f"{name}/")
        elif isinstance(value, Labeled):
            for label, v in value.named_scalars():
                yield f"{name}/{label}", v
        elif isinstance(value, clu_metrics.Metric):
            yield name, float(value.compute())
