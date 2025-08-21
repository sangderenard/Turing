"""
Demo: MNIST classification using AbstractTensor and abstract_nn
Backend-agnostic, works with torch or numpy backends.
"""
import os
import gzip
import urllib.request
import numpy as np
from PIL import Image

from ..abstraction import AbstractTensor
from .core import Linear, Model, RectConv2d, MaxPool2d, Flatten
from .activations import Identity, ReLU
from .losses import CrossEntropyLoss
from .optimizer import Adam
from .train import train_loop
from .utils import set_seed
# --- Robust MNIST downloader (mirrors + checksums) ---
import hashlib
from urllib.error import URLError, HTTPError
MNIST_DIR = os.path.join(os.path.dirname(__file__), "mnist_data")
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://raw.githubusercontent.com/fgnt/mnist/master",
    # keep the original as last resort:
    "https://yann.lecun.com/exdb/mnist",
]

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_mnist():
    for fname, md5 in MNIST_FILES.items():
        out_path = os.path.join(MNIST_DIR, fname)
        # skip if present and hash matches
        if os.path.exists(out_path):
            try:
                if _md5(out_path) == md5:
                    continue
            except Exception:
                pass  # fall through and re-download

        # try mirrors
        last_err = None
        for base in MIRRORS:
            url = f"{base}/{fname}"
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, out_path)
                if _md5(out_path) == md5:
                    break  # success
                else:
                    print(f"Checksum mismatch for {fname}, trying next mirror...")
            except (HTTPError, URLError) as e:
                last_err = e
                print(f"Failed from {base}: {e}")
        else:
            raise RuntimeError(f"Could not fetch {fname} from any mirror") from last_err

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic
        n = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0
        return data

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(4)  # magic
        n = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def get_mnist():
    download_mnist()
    X_train = load_images(os.path.join(MNIST_DIR, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte.gz'))
    X_test = load_images(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte.gz'))
    y_test = load_labels(os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test

# --- Main demo ---
def main():
    set_seed(42)
    X_train, y_train, X_test, y_test = get_mnist()
    # Use a small subset for demo speed
    N = 10000
    X = X_train[:N]
    y = y_train[:N]
    # Convert to AbstractTensor (torch or numpy backend)
    like = AbstractTensor.get_tensor(X)
    X_tensor = like.ensure_tensor(X)
    y_tensor = like.ensure_tensor(y.reshape(-1, 1))
    # CNN model
    model = Model([
        RectConv2d(1, 8, 3, padding=1, like=like),
        RectConv2d(8, 16, 3, padding=1, like=like),
        MaxPool2d(2, stride=2, like=like),
        RectConv2d(16, 32, 3, padding=1, like=like),
        MaxPool2d(2, stride=2, like=like),
        Flatten(like=like),
        Linear(32 * 7 * 7, 64, like=like),
        Linear(64, 10, like=like)
    ], [ReLU(), ReLU(), None, ReLU(), None, None, ReLU(), None])
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    print("Training...")
    train_loop(model, loss_fn, optimizer, X_tensor, y_tensor, epochs=1000, log_every=1)
    # Evaluate
    logits = model.forward(X_tensor)
    preds = logits.argmax(dim=1)
    acc = (preds.reshape(-1) == y_tensor.reshape(-1)).sum().item() / N
    print(f"Train accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
