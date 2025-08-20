"""
Demo: CIFAR-10 classification using AbstractTensor and abstract_nn
Backend-agnostic, works with torch or numpy backends.
"""
import os
import tarfile
import urllib.request
import numpy as np
from PIL import Image

from ..abstraction import AbstractTensor
from .core import Linear, Model
from .activations import Identity
from .losses import CrossEntropyLoss
from .optimizer import Adam
from .train import train_loop
from .utils import set_seed


# --- Robust CIFAR-10 downloader (mirrors + checksums) ---
import hashlib
from urllib.error import URLError, HTTPError
CIFAR10_DIR = os.path.join(os.path.dirname(__file__), "cifar10_data")
os.makedirs(CIFAR10_DIR, exist_ok=True)
CIFAR10_FILE = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_MIRRORS = [
    "https://www.cs.toronto.edu/~kriz",
    "https://www.cs.toronto.edu/~kriz",
    "https://www.cs.toronto.edu/~kriz",  # fallback: all official, but you can add more
]

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_cifar10():
    out_path = os.path.join(CIFAR10_DIR, CIFAR10_FILE)
    # skip if present and hash matches
    if os.path.exists(out_path):
        try:
            if _md5(out_path) == CIFAR10_MD5:
                pass
            else:
                print(f"Checksum mismatch for {CIFAR10_FILE}, will re-download...")
        except Exception:
            print(f"Could not check checksum for {CIFAR10_FILE}, will re-download...")
    else:
        last_err = None
        for base in CIFAR10_MIRRORS:
            url = f"{base}/{CIFAR10_FILE}"
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, out_path)
                if _md5(out_path) == CIFAR10_MD5:
                    break  # success
                else:
                    print(f"Checksum mismatch for {CIFAR10_FILE}, trying next mirror...")
            except (HTTPError, URLError) as e:
                last_err = e
                print(f"Failed from {base}: {e}")
        else:
            raise RuntimeError(f"Could not fetch {CIFAR10_FILE} from any mirror") from last_err

    # Extract if not already
    extract_dir = os.path.join(CIFAR10_DIR, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10...")
        with tarfile.open(out_path, "r:gz") as tar:
            tar.extractall(CIFAR10_DIR)

def load_batch(filename):
    import pickle
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data']
        y = dict[b'labels']
        X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        X = np.transpose(X, (0, 2, 3, 1))  # (N, H, W, C)
        X = X.reshape(-1, 32*32*3)
        y = np.array(y, dtype=np.int64)
        return X, y

def get_cifar10():
    download_cifar10()
    base = os.path.join(CIFAR10_DIR, "cifar-10-batches-py")
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(base, f"data_batch_{i}"))
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_batch(os.path.join(base, "test_batch"))
    return X_train, y_train, X_test, y_test

# --- Main demo ---
def main():
    set_seed(42)
    X_train, y_train, X_test, y_test = get_cifar10()
    # Use a small subset for demo speed
    N = 10000
    X = X_train[:N]
    y = y_train[:N]
    # Convert to AbstractTensor (torch or numpy backend)
    like = AbstractTensor.get_tensor(X)
    X_tensor = like.ensure_tensor(X)
    y_tensor = like.ensure_tensor(y.reshape(-1, 1))
    # Model: 3072 -> 512 -> 256 -> 10
    model = Model([
        Linear(3072, 512, like=like),
        Linear(512, 256, like=like),
        Linear(256, 10, like=like)
    ], [Identity(), Identity(), None])
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    print("Training...")
    train_loop(model, loss_fn, optimizer, X_tensor, y_tensor, epochs=10, log_every=1)
    # Evaluate
    logits = model.forward(X_tensor)
    preds = logits.argmax(dim=1)
    acc = (preds.reshape(-1) == y_tensor.reshape(-1)).sum().item() / N
    print(f"Train accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
