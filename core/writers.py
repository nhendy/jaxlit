# The generated tensorboard protos define their message classes at runtime,
# so the type checker cannot see `summary_pb2.Summary` and friends.
# pyright: reportAttributeAccessIssue=false
"""Metric writers. Values are `(name, value)` pairs, never dicts.

Why not `clu.metric_writers`: importing anything under it runs the package
`__init__`, which imports TensorFlow, and TensorFlow and Grain segfault when
loaded into the same process. The TensorBoard writer here uses the
`tensorboard` package's own event-file writer and protos, which need no TF.
"""

import io
import pathlib
import time
from collections.abc import Iterable
from typing import Any, Protocol

import attrs
import numpy as np
from absl import logging
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

type Scalars = Iterable[tuple[str, float]]
# `(name, image)` with image `(h, w, c)`, c in {1, 3}, float in [0, 1] or uint8.
type Images = Iterable[tuple[str, np.ndarray]]


class Figure(Protocol):
    """An interactive figure -- in practice a `plotly.graph_objects.Figure`.

    Structural, so `core` does not import plotly: anything that serializes to
    plotly JSON qualifies.
    """

    def to_json(self, *args: Any, **kwargs: Any) -> Any: ...


type Figures = Iterable[tuple[str, Figure]]


class MetricWriter(Protocol):
    def write_scalars(self, step: int, scalars: Scalars) -> None: ...

    def write_images(self, step: int, images: Images) -> None: ...

    def write_figures(self, step: int, figures: Figures) -> None:
        """Interactive figures. Writers that cannot show one skip it."""
        ...

    def flush(self) -> None: ...


@attrs.frozen
class LoggingWriter:
    def write_scalars(self, step: int, scalars: Scalars) -> None:
        line = ", ".join(f"{name}={value:.6g}" for name, value in scalars)
        logging.info("[%d] %s", step, line)

    def write_images(self, step: int, images: Images) -> None:
        for name, image in images:
            logging.info("[%d] image %s %s", step, name, image.shape)

    def write_figures(self, step: int, figures: Figures) -> None:
        for name, _ in figures:
            logging.info("[%d] figure %s", step, name)

    def flush(self) -> None:
        return None


@attrs.frozen(kw_only=True)
class ScalarRecord:
    step: int
    name: str
    value: float


@attrs.frozen(kw_only=True)
class ImageRecord:
    step: int
    name: str
    image: np.ndarray = attrs.field(eq=False)


@attrs.frozen(kw_only=True)
class FigureRecord:
    step: int
    name: str
    figure: Figure = attrs.field(eq=False)


@attrs.frozen
class InMemoryWriter:
    """Records everything it is given. For tests.

    Frozen like every other static object; the record lists are the one piece
    of mutable state, owned here so nothing else holds it.
    """

    scalars: list[ScalarRecord] = attrs.field(factory=list)
    images: list[ImageRecord] = attrs.field(factory=list)
    figures: list[FigureRecord] = attrs.field(factory=list)

    def write_scalars(self, step: int, scalars: Scalars) -> None:
        self.scalars.extend(
            ScalarRecord(step=step, name=n, value=v) for n, v in scalars
        )

    def write_images(self, step: int, images: Images) -> None:
        self.images.extend(ImageRecord(step=step, name=n, image=i) for n, i in images)

    def write_figures(self, step: int, figures: Figures) -> None:
        self.figures.extend(
            FigureRecord(step=step, name=n, figure=f) for n, f in figures
        )

    def flush(self) -> None:
        return None


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _png(image: np.ndarray) -> bytes:

    pixels = to_uint8(image)
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@attrs.frozen
class TensorBoardWriter:
    """Event files TensorBoard reads, written without TensorFlow."""

    writer: EventFileWriter

    @classmethod
    def create(cls, logdir: pathlib.Path) -> "TensorBoardWriter":
        logdir.mkdir(parents=True, exist_ok=True)
        return cls(EventFileWriter(str(logdir)))

    def _write(self, step: int, values: list[summary_pb2.Summary.Value]) -> None:
        summary = summary_pb2.Summary(value=values)
        self.writer.add_event(
            event_pb2.Event(wall_time=time.time(), step=step, summary=summary)
        )

    def write_scalars(self, step: int, scalars: Scalars) -> None:
        self._write(
            step,
            [summary_pb2.Summary.Value(tag=n, simple_value=v) for n, v in scalars],
        )

    def write_images(self, step: int, images: Images) -> None:
        values = []
        for name, image in images:
            h, w, c = image.shape
            encoded = summary_pb2.Summary.Image(
                height=h, width=w, colorspace=c, encoded_image_string=_png(image)
            )
            values.append(summary_pb2.Summary.Value(tag=name, image=encoded))
        self._write(step, values)

    def write_figures(self, step: int, figures: Figures) -> None:
        # TensorBoard has no interactive figure panel, and rasterizing plotly
        # needs kaleido and a headless Chrome. Images are the TensorBoard route.
        return None

    def flush(self) -> None:
        self.writer.flush()


@attrs.frozen
class MultiWriter:
    writers: tuple[MetricWriter, ...]

    def write_scalars(self, step: int, scalars: Scalars) -> None:
        scalars = tuple(scalars)
        for writer in self.writers:
            writer.write_scalars(step, scalars)

    def write_images(self, step: int, images: Images) -> None:
        images = tuple(images)
        for writer in self.writers:
            writer.write_images(step, images)

    def write_figures(self, step: int, figures: Figures) -> None:
        figures = tuple(figures)
        for writer in self.writers:
            writer.write_figures(step, figures)

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


def default_writer(logdir: pathlib.Path) -> MultiWriter:
    """Log lines plus TensorBoard event files under `logdir`."""
    return MultiWriter((LoggingWriter(), TensorBoardWriter.create(logdir)))
