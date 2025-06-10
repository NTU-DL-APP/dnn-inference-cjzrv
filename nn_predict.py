"""
Fashion‑MNIST – **train + export** vs **NumPy‑only inference**
=============================================================
This module cleanly separates **training** and **inference** so that *importing*
`fashion_mnist.py` never triggers a costly re‑training step.

* Run `python fashion_mnist.py train` (or just `python fashion_mnist.py`) to
  **train** a CNN and export:

    • `fashion_mnist.h5` – full Keras model
    • `fashion_mnist.npz` – weights (NumPy)
    • `fashion_mnist.json` – architecture (Keras JSON)

* In any other script do:

```python
from fashion_mnist import nn_inference, relu, softmax
# load json / npz & call nn_inference(...)
```

and you will **only** get the lightweight NumPy inference helpers – *no TensorFlow,
no training loop*.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np

# =============================================================
# ── NumPy‑only helpers (safe on import) ───────────────────────
# =============================================================

def relu(x: np.ndarray) -> np.ndarray:  # noqa: D401 (simple)
    """ReLU activation: \max(0, x)."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically‑stable softmax along the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def dense(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b


# -------------------------------------------------------------
# Keras‑>minimal spec helper (called **only** during export)
# -------------------------------------------------------------

def _make_dense_flatten_spec(tf_model):  # type: ignore
    spec = []
    import tensorflow as tf  # local import to avoid TF unless needed
    for lyr in tf_model.layers:
        if isinstance(lyr, tf.keras.layers.Flatten):
            spec.append({"class_name": "Flatten", "weights": []})
        elif isinstance(lyr, tf.keras.layers.Dense):
            wnames = [w.name for w in lyr.weights]
            spec.append({
                "class_name": "Dense",
                "config": {"activation": lyr.activation.__name__},
                "weights": wnames,
            })
    return spec


# -------------------------------------------------------------
# Pure‑NumPy forward pass (inference) – public API
# -------------------------------------------------------------

def nn_forward_h5(model_arch: list[dict], weights_map: dict[str, np.ndarray], data: np.ndarray) -> np.ndarray:  # noqa: D401
    x = data
    for layer in model_arch:
        ltype = layer["class_name"]
        if ltype == "Flatten":
            x = flatten(x)
            continue
        if ltype == "Dense":
            W, b = (weights_map[wn] for wn in layer["weights"])
            x = dense(x, W, b)
            act = layer["config"].get("activation", "linear")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)
    return x


def nn_inference(model_arch: list[dict], weights_map: dict[str, np.ndarray], data: np.ndarray) -> np.ndarray:  # noqa: D401
    """Wrapper expected by evaluation harness."""
    return nn_forward_h5(model_arch, weights_map, data)

# =============================================================
# ── Training & export (executed *only* when run as script) ────
# =============================================================


def _train_and_export():  # noqa: C901 – keep in single function for clarity
    import tensorflow as tf  # local import keeps TF out of inference‑only runs

    # 1. Hyper‑parameters
    BATCH   = 256
    EPOCHS  = 20  # 30→≈95 % if you have time
    LR      = 0.001
    NAME    = "fashion_mnist"

    # 2. Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., None]
    x_test  = x_test[..., None]

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    AUTO = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(10_000).batch(BATCH)
                .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO)
                .prefetch(AUTO))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(BATCH).prefetch(AUTO))

    # 3. Model
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x);  x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x);  x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inp, out, name="fashion_mnist_cnn")

    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau("val_accuracy", factor=0.5, patience=3, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.EarlyStopping("val_accuracy", patience=6, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=cbs)

    print("\nTest set eval:"); model.evaluate(test_ds, verbose=2)

    # 4. Export
    Path(f"{NAME}.h5").parent.mkdir(parents=True, exist_ok=True)
    model.save(f"{NAME}.h5", include_optimizer=False)
    np.savez(f"{NAME}.npz", *model.get_weights())

    # Minimal spec for NumPy path
    spec = _make_dense_flatten_spec(model)
    with open(f"{NAME}.json", "w") as f:
        json.dump(spec, f)

    print("\nSaved:", *(str(p) for p in [f"{NAME}.h5", f"{NAME}.npz", f"{NAME}.json"]))

# =============================================================
# ── CLI entry‑point ───────────────────────────────────────────
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion‑MNIST trainer / exporter")
    parser.add_argument("mode", nargs="?", default="train", choices=["train"], help="currently only 'train'")
    args = parser.parse_args()

    if args.mode == "train":
        _train_and_export()
