```python
uv run python -c
"
import zipfile, numpy as np
from huggingface_hub import hf_hub_download
pt = hf_hub_download(repo_id='BAAI/bge-m3', filename='colbert_linear.pt')
with zipfile.ZipFile(pt) as zf:
    for n in zf.namelist():
        info = zf.getinfo(n)
        print(f'{n}: {info.file_size} bytes (compressed: {info.compress_size})')
    data_files = sorted(n for n in zf.namelist() if '/data/' in n and not n.endswith('/'))
    for df in data_files:
        with zf.open(df) as f:
            raw = f.read()
        print(f'\n{df}: {len(raw)} raw bytes')
        for dtype, name in [(np.float16, 'fp16'), (np.float32, 'fp32')]:
            arr = np.frombuffer(raw, dtype=dtype)
            print(f'  as {name}: {arr.shape[0]} elements')
"
```
