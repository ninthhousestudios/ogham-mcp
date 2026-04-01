```bash
cd /workspace/ogham-mcp && git pull && uv run python -c "
from ogham.onnx_embedder import encode_query_colbert
vecs = encode_query_colbert('test query')
print(f'Shape: {vecs.shape}')
print(f'Norm: {(vecs[0] ** 2).sum() ** 0.5:.4f}')
"
```
