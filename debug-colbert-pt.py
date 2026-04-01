"""Check whether the ONNX colbert_vectors output matches the PyTorch
reference (with colbert_linear applied)."""
import zipfile
import numpy as np
from huggingface_hub import hf_hub_download

# Load the colbert_linear weight and bias from colbert_linear.pt
pt = hf_hub_download(repo_id="BAAI/bge-m3", filename="colbert_linear.pt")
with zipfile.ZipFile(pt) as zf:
    with zf.open("colbert_linear/data/0") as f:
        W = np.frombuffer(f.read(), dtype=np.float16).reshape(1024, 1024).astype(np.float32)
    with zf.open("colbert_linear/data/1") as f:
        b = np.frombuffer(f.read(), dtype=np.float16).reshape(1024).astype(np.float32)

print(f"colbert_linear weight: {W.shape}, bias: {b.shape}")

# Get the ONNX model's colbert output (raw 1024-dim)
from ogham.onnx_embedder import _get_model
tokenizer, session = _get_model()
encoded = tokenizer.encode("test query")
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
onnx_colbert = outputs[2][0].astype(np.float32)  # [n_tokens, 1024]
print(f"ONNX colbert_vectors: {onnx_colbert.shape}")

# Apply the linear projection manually: projected = x @ W.T + b
projected = onnx_colbert @ W.T + b
print(f"After projection: {projected.shape}")

# L2 normalize both
onnx_norm = onnx_colbert / np.maximum(np.linalg.norm(onnx_colbert, axis=1, keepdims=True), 1e-12)
proj_norm = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)

# Compare: if ONNX output already has the projection, these should be similar
cos_sim = np.sum(onnx_norm * proj_norm, axis=1)
print(f"Cosine similarity (onnx vs projected), per token: {cos_sim}")
print(f"Mean cosine sim: {cos_sim.mean():.4f}")
print()
print("If cosine ~1.0: ONNX already includes the projection (no fix needed)")
print("If cosine <<1.0: ONNX is raw hidden states, projection is needed")
