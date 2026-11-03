"""MNIST as typed batches through a Grain pipeline.

Loaded from the Keras `mnist.npz` mirror rather than tensorflow-datasets: tfds
imports TensorFlow, which segfaults alongside Grain in one process.
"""

import enum
import pathlib
import urllib.request

import attrs
import eincheck
import grain
import numpy as np
from flax import struct

from core.distributed import Shard
from core.types import HostOrDeviceArray

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
# Under the repo's git-ignored /datasets/.
DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "mnist"


@eincheck.check_data(image="*n b 28 28 1", label="*n b")
@struct.dataclass
class MnistBatch:
    # Float32 in [0, 1].
    image: HostOrDeviceArray
    label: HostOrDeviceArray


class Split(enum.Enum):
    TRAIN = enum.auto()
    TEST = enum.auto()


@attrs.frozen(kw_only=True)
class MnistArrays:
    images: np.ndarray = attrs.field(eq=False)
    labels: np.ndarray = attrs.field(eq=False)

    def __len__(self) -> int:
        return len(self.labels)


def download(directory: pathlib.Path = DEFAULT_DIR) -> pathlib.Path:
    path = directory / "mnist.npz"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def load(split: Split, directory: pathlib.Path = DEFAULT_DIR) -> MnistArrays:
    with np.load(download(directory)) as npz:
        match split:
            case Split.TRAIN:
                images, labels = npz["x_train"], npz["y_train"]
            case Split.TEST:
                images, labels = npz["x_test"], npz["y_test"]
    return MnistArrays(
        images=(images.astype(np.float32) / 255.0)[..., None],
        labels=labels.astype(np.int32),
    )


@attrs.frozen
class Gather:
    """Grain map: a batch of indices -> a `MnistBatch`."""

    arrays: MnistArrays

    def __call__(self, indices: np.ndarray) -> MnistBatch:
        return MnistBatch(
            image=self.arrays.images[indices], label=self.arrays.labels[indices]
        )


def batches(
    arrays: MnistArrays,
    *,
    batch_size: int,
    seed: int,
    training: bool,
    shard: Shard | None = None,
) -> grain.IterDataset[MnistBatch]:
    """Training: shuffled, repeated forever, full batches. Eval: one ordered pass.

    The result is re-iterable, which `core.callbacks.Evaluate` relies on.
    `batch_size` is per process; each reads its own `shard` (by default, one
    slice per process).
    """
    shard = Shard.current() if shard is None else shard
    ds = grain.MapDataset.range(len(arrays))[shard.index :: shard.count]
    if training:
        ds = ds.shuffle(seed=seed).repeat()
    ds = ds.batch(batch_size, drop_remainder=training).map(Gather(arrays))
    return ds.to_iter_dataset()
