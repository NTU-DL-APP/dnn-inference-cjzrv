import numpy as np
import json

# === Activation functions ===
def relu(x):
    # ReLU(x) = max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # Softmax for numerical stability
    # subtract max along axis=1 to prevent overflow
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for i, layer in enumerate(model_arch["config"]["layers"]):
        ltype = layer['class_name']
        cfg = layer['config']
        wnames = [f'param_{2 * i}', f'param_{2 * i + 1}']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
