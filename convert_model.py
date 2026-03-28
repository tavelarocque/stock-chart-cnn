"""
convert_model.py
----------------
Converts the trained Keras model to ONNX format for lightweight deployment.
Only needs to be run once locally. Requires tf2onnx.

Usage:
    pip3 install tf2onnx onnx
    python3 convert_model.py
"""

import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

KERAS_MODEL_PATH = "results/models/best_model.keras"
ONNX_MODEL_PATH  = "results/models/model.onnx"

print("📦 Loading Keras model...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
model.summary()

print("\n🔄 Converting to ONNX...")
input_signature = [tf.TensorSpec([None, 64, 64, 3], tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

onnx.save(onnx_model, ONNX_MODEL_PATH)
print(f"✅ ONNX model saved to: {ONNX_MODEL_PATH}")

# Quick sanity check
import onnxruntime as ort
session = ort.InferenceSession(ONNX_MODEL_PATH)
dummy = np.random.rand(1, 64, 64, 3).astype(np.float32)
out = session.run(None, {"input": dummy})
print(f"✅ Test inference output: {out[0][0][0]:.4f} (should be a number between 0 and 1)")
