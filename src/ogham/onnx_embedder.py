"""ONNX BGE-M3 embedding provider.

Produces dense + sparse + ColBERT vectors in a single model pass using the
yuniko-software/bge-m3-onnx model with HuggingFace's `tokenizers` library.

All heavy imports (onnxruntime, tokenizers, numpy) are lazy — this module
is safe to import even when the ONNX deps aren't installed.
"""

from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# XLM-RoBERTa special tokens: [PAD]=1, [UNK]=3, [CLS]=0, [SEP]=2
_SPECIAL_TOKEN_IDS = frozenset({0, 1, 2, 3})

# BGE-M3 vocabulary size (XLM-RoBERTa)
VOCAB_SIZE = 250002


@dataclass
class OnnxResult:
    """Result from a single ONNX BGE-M3 forward pass."""

    dense: list[float]
    sparse: dict[int, float]
    colbert: bytes | None = None  # packed float32: 8-byte header (n_tokens, dim) + data


# ── Singleton model holder ────────────────────────────────────────────

_tokenizer = None
_session = None
_colbert_linear = None  # lazy-loaded ColBERT projection weight [128, 1024]
_model_lock = threading.Lock()


def _get_model(model_path: str | None = None):
    """Lazy-load the ONNX session and tokenizer (singleton, thread-safe)."""
    global _tokenizer, _session
    if _session is not None:
        return _tokenizer, _session

    with _model_lock:
        # Double-check after acquiring lock
        if _session is not None:
            return _tokenizer, _session

        import onnxruntime as ort
        from tokenizers import Tokenizer

        if model_path is None:
            model_path = str(Path.home() / ".cache" / "ogham" / "bge-m3-onnx" / "bge_m3_model.onnx")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run 'ogham download-model bge-m3' to download it."
            )

        logger.info("Loading tokenizer for BAAI/bge-m3...")
        _tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")
        _tokenizer.enable_truncation(max_length=8192)
        _tokenizer.no_padding()

        logger.info("Loading ONNX model from %s...", model_path)
        options = ort.SessionOptions()
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = False  # release memory between inferences
        options.log_severity_level = 2  # WARNING
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        # Use GPU if available, otherwise CPU
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        _session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
        )
        active = _session.get_providers()
        logger.info("ONNX BGE-M3 model loaded (providers: %s).", active)

        # If CUDA was requested but didn't load, warn loudly
        if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in active:
            logger.warning("CUDAExecutionProvider was available but failed to load!")
        return _tokenizer, _session


def _get_colbert_linear():
    """Lazy-load the colbert_linear projection weight from BAAI/bge-m3.

    The ONNX export (yuniko-software/bge-m3-onnx) omits the colbert_linear
    layer, so its third output is raw 1024-dim hidden states.  We download
    just the projection weight from HuggingFace and apply it in Python:
        projected = token_vecs @ W.T   (then L2-normalize)
    """
    global _colbert_linear
    if _colbert_linear is not None:
        return _colbert_linear

    with _model_lock:
        if _colbert_linear is not None:
            return _colbert_linear

        import numpy as np

        weight_path = Path.home() / ".cache" / "ogham" / "bge-m3-onnx" / "colbert_linear.npy"

        if not weight_path.exists():
            logger.info("Downloading colbert_linear weight from BAAI/bge-m3...")
            _download_colbert_linear(weight_path)

        _colbert_linear = np.load(weight_path)  # [128, 1024] float32
        logger.info("ColBERT projection loaded: %s", _colbert_linear.shape)
        return _colbert_linear


def _download_colbert_linear(dest: Path):
    """Extract colbert_linear.weight from BAAI/bge-m3 safetensors checkpoint."""
    import numpy as np

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download the ColBERT projection weight. "
            "Install it with: pip install huggingface_hub"
        )

    try:
        from safetensors.numpy import load_file

        st_path = hf_hub_download(repo_id="BAAI/bge-m3", filename="model.safetensors")
        tensors = load_file(st_path)
        key = "colbert_linear.weight"
        if key not in tensors:
            raise KeyError(f"{key} not found. Keys: {list(tensors.keys())[:10]}...")
        weight = tensors[key].astype(np.float32)  # [128, 1024]
    except ImportError:
        raise ImportError(
            "safetensors is required to load the ColBERT projection weight. "
            "Install it with: pip install safetensors"
        )

    np.save(dest, weight)
    logger.info("Saved colbert_linear weight to %s (%s)", dest, weight.shape)


def _apply_colbert_projection(token_vecs):
    """Project 1024-dim hidden states to 128-dim ColBERT space + L2-normalize."""
    import numpy as np

    W = _get_colbert_linear()  # [128, 1024]
    projected = token_vecs @ W.T  # [n_tokens, 128]
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (projected / norms).astype(np.float32)


# ── Encoding ──────────────────────────────────────────────────────────


def encode(text: str, model_path: str | None = None, *, include_colbert: bool = False) -> OnnxResult:
    """Encode a single text, returning dense + sparse + optional ColBERT vectors."""
    import numpy as np

    tokenizer, session = _get_model(model_path)

    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    dense_embeddings, sparse_weights, colbert_vectors = outputs

    # Dense: already L2-normalized by the model export
    dense = dense_embeddings[0].tolist()

    # Sparse: per-token max weight, skip specials
    sparse: dict[int, float] = {}
    for i, token_id in enumerate(encoded.ids):
        if encoded.attention_mask[i] == 1 and token_id not in _SPECIAL_TOKEN_IDS:
            weight = float(np.max(sparse_weights[0, i]))
            if weight > 0:
                sparse[token_id] = max(sparse.get(token_id, 0), weight)

    # ColBERT: per-token vectors — the ONNX export already excludes one
    # special token position, so take the full output as-is
    colbert_bytes = None
    if include_colbert:
        token_vecs = _apply_colbert_projection(colbert_vectors[0])
        colbert_bytes = pack_colbert(token_vecs)

    return OnnxResult(dense=dense, sparse=sparse, colbert=colbert_bytes)


def encode_batch(
    texts: list[str], model_path: str | None = None, *, include_colbert: bool = False
) -> list[OnnxResult]:
    """Encode multiple texts sequentially.

    We disable padding and loop instead of batching because sparse weight
    extraction needs per-token attention masks without padding noise.
    Batching with padding would inflate sparse weights for pad positions.
    """
    return [encode(text, model_path, include_colbert=include_colbert) for text in texts]


# ── Sparse format conversion ─────────────────────────────────────────


# ── ColBERT format conversion ─────────────────────────────────────


_DEFAULT_POOL_FACTOR = 2


def pool_colbert(token_vectors, pool_factor: int = _DEFAULT_POOL_FACTOR):
    """Compress ColBERT token vectors via hierarchical clustering + mean pooling.

    Clusters similar token vectors within a document using Ward's linkage on
    cosine distances, then averages vectors within each cluster. A pool_factor
    of N reduces token count to ceil(n_tokens / N).

    Based on Answer.AI's ColBERT Token Pooling research:
    - Factor 2: 100.6% of baseline quality (slight improvement)
    - Factor 3: 99% of baseline
    - Factor 4: 97% of baseline

    Returns pooled [n_clusters, dim] array.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n_tokens, dim = token_vectors.shape
    max_clusters = max(1, n_tokens // pool_factor)

    if n_tokens <= max_clusters:
        return token_vectors

    # Cosine distance matrix
    norms = np.linalg.norm(token_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = token_vectors / norms
    sim = normed @ normed.T
    np.clip(sim, -1, 1, out=sim)
    dist = 1 - sim

    # Convert to condensed form for scipy
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    # Ward's linkage needs non-negative distances
    np.maximum(condensed, 0, out=condensed)

    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=max_clusters, criterion="maxclust")

    # Mean-pool within clusters
    pooled = np.zeros((max_clusters, dim), dtype=token_vectors.dtype)
    for c in range(1, max_clusters + 1):
        mask = labels == c
        if mask.any():
            pooled[c - 1] = token_vectors[mask].mean(axis=0)

    return pooled


def pack_colbert(token_vectors, pool_factor: int = _DEFAULT_POOL_FACTOR) -> bytes:
    """Pool and pack ColBERT vectors into bytes using float16.

    Applies token pooling (hierarchical clustering + mean pool) to reduce
    vector count by pool_factor, then stores as float16.

    Format: 4-byte n_tokens (uint32) + 4-byte dim (uint32) + flat float16 data.
    """
    import numpy as np

    pooled = pool_colbert(token_vectors, pool_factor)
    n_tokens, dim = pooled.shape
    return struct.pack("II", n_tokens, dim) + pooled.astype(np.float16).tobytes()


def pack_colbert_raw(token_vectors) -> bytes:
    """Pack ColBERT vectors WITHOUT pooling, stored as float32.

    For benchmarking: preserves full token-level detail for quality comparison
    against pooled float16. Much larger (~8x) than pack_colbert().

    Format: 4-byte n_tokens (uint32) + 4-byte dim (uint32) + flat float32 data.
    """
    import numpy as np

    vecs = token_vectors.astype(np.float32)
    n_tokens, dim = vecs.shape
    return struct.pack("II", n_tokens, dim) + vecs.tobytes()


def unpack_colbert(data: bytes):
    """Unpack bytes into a [n_tokens, dim] float32 numpy array.

    Detects float16 vs float32 format from data size and upscales float16
    to float32 for computation.
    """
    import numpy as np

    n_tokens, dim = struct.unpack("II", data[:8])
    expected_f16 = n_tokens * dim * 2
    expected_f32 = n_tokens * dim * 4
    payload = data[8:]

    if len(payload) == expected_f16:
        return np.frombuffer(payload, dtype=np.float16).reshape(n_tokens, dim).astype(np.float32)
    elif len(payload) == expected_f32:
        return np.frombuffer(payload, dtype=np.float32).reshape(n_tokens, dim)
    else:
        raise ValueError(
            f"ColBERT payload size {len(payload)} doesn't match "
            f"float16 ({expected_f16}) or float32 ({expected_f32}) "
            f"for shape ({n_tokens}, {dim})"
        )


def pack_colbert_int8_row(token_vectors) -> bytes:
    """Pack ColBERT vectors as int8 with per-row scale factors.

    Each token vector gets its own scale: scale_i = max(|row_i|) / 127.
    Quantized value = round(value / scale), clamped to [-127, 127].
    Caller is responsible for pooling first if desired.

    Format: 4-byte n_tokens + 4-byte dim + 1-byte tag (0x02) +
            n_tokens × float32 scales + n_tokens × dim × int8 data.
    """
    import numpy as np

    vecs = token_vectors.astype(np.float32)
    n_tokens, dim = vecs.shape

    # Per-row scale factors
    row_max = np.max(np.abs(vecs), axis=1)
    scales = np.maximum(row_max / 127.0, 1e-8).astype(np.float32)

    # Quantize
    quantized = np.round(vecs / scales[:, None]).clip(-127, 127).astype(np.int8)

    header = struct.pack("IIB", n_tokens, dim, 0x02)
    return header + scales.tobytes() + quantized.tobytes()


def pack_colbert_int8_channel(token_vectors) -> bytes:
    """Pack ColBERT vectors as int8 with per-channel (per-dimension) scale factors.

    Each dimension gets its own scale: scale_j = max(|col_j|) / 127.
    Better fidelity when dimensions have varying magnitudes.
    Caller is responsible for pooling first if desired.

    Format: 4-byte n_tokens + 4-byte dim + 1-byte tag (0x03) +
            dim × float32 scales + n_tokens × dim × int8 data.
    """
    import numpy as np

    vecs = token_vectors.astype(np.float32)
    n_tokens, dim = vecs.shape

    # Per-channel scale factors
    col_max = np.max(np.abs(vecs), axis=0)
    scales = np.maximum(col_max / 127.0, 1e-8).astype(np.float32)

    # Quantize
    quantized = np.round(vecs / scales[None, :]).clip(-127, 127).astype(np.int8)

    header = struct.pack("IIB", n_tokens, dim, 0x03)
    return header + scales.tobytes() + quantized.tobytes()


def unpack_colbert_int8_row(data: bytes):
    """Unpack int8 per-row quantized ColBERT vectors to float32."""
    import numpy as np

    n_tokens, dim, tag = struct.unpack("IIB", data[:9])
    payload = data[9:]

    scales_size = n_tokens * 4
    scales = np.frombuffer(payload[:scales_size], dtype=np.float32)
    quantized = np.frombuffer(payload[scales_size:], dtype=np.int8).reshape(n_tokens, dim)

    return quantized.astype(np.float32) * scales[:, None]


def unpack_colbert_int8_channel(data: bytes):
    """Unpack int8 per-channel quantized ColBERT vectors to float32."""
    import numpy as np

    n_tokens, dim, tag = struct.unpack("IIB", data[:9])
    payload = data[9:]

    scales_size = dim * 4
    scales = np.frombuffer(payload[:scales_size], dtype=np.float32)
    quantized = np.frombuffer(payload[scales_size:], dtype=np.int8).reshape(n_tokens, dim)

    return quantized.astype(np.float32) * scales[None, :]


def unpack_colbert_any(data: bytes):
    """Auto-detect format and unpack ColBERT vectors to float32.

    Handles all formats: f32 (raw), f16 (pooled), int8-row (tag 0x02),
    int8-channel (tag 0x03).

    Detection uses both the tag byte AND expected payload size to avoid
    false positives — byte 8 in legacy f16 data is a float16 value that
    could coincidentally equal 0x02 or 0x03.

    Returns: numpy array [n_tokens, dim] float32.
    """
    if len(data) >= 9:
        n_tokens, dim, tag = struct.unpack("IIB", data[:9])
        if tag == 0x02:
            expected = 9 + n_tokens * 4 + n_tokens * dim  # scales + int8 data
            if len(data) == expected:
                return unpack_colbert_int8_row(data)
        elif tag == 0x03:
            expected = 9 + dim * 4 + n_tokens * dim  # scales + int8 data
            if len(data) == expected:
                return unpack_colbert_int8_channel(data)

    # Fall through to legacy 8-byte header (f16/f32 size detection)
    return unpack_colbert(data)


def repack_colbert(raw_f32_vectors, pool_factor: int, precision: str) -> bytes:
    """One-stop function: pool at factor N, pack at precision P.

    Args:
        raw_f32_vectors: [n_tokens, dim] float32 array (unpooled source of truth).
        pool_factor: Token reduction factor (1 = no pooling).
        precision: One of 'f32', 'f16', 'int8_row', 'int8_channel'.

    Returns: Packed bytes ready for DB storage.
    """
    import numpy as np

    pooled = pool_colbert(raw_f32_vectors, pool_factor) if pool_factor > 1 else raw_f32_vectors.astype(np.float32)

    if precision == "f32":
        return pack_colbert_raw(pooled)
    elif precision == "f16":
        n_tokens, dim = pooled.shape
        return struct.pack("II", n_tokens, dim) + pooled.astype(np.float16).tobytes()
    elif precision == "int8_row":
        return pack_colbert_int8_row(pooled)
    elif precision == "int8_channel":
        return pack_colbert_int8_channel(pooled)
    else:
        raise ValueError(f"Unknown precision: {precision!r}. Use 'f32', 'f16', 'int8_row', or 'int8_channel'.")


# ── MaxSim scoring ───────────────────────────────────────────────


def maxsim(query_vectors, doc_vectors) -> float:
    """ColBERT MaxSim: for each query token, find max similarity to any doc token, sum.

    Args:
        query_vectors: [n_q, dim] float32 — unpooled query token vectors.
        doc_vectors: [n_d, dim] float32 — pooled (or raw) document token vectors.

    Returns:
        Scalar MaxSim score (higher = more relevant).
    """
    sim = query_vectors @ doc_vectors.T  # [n_q, n_d]
    return float(sim.max(axis=1).sum())


def encode_query_colbert(text: str, model_path: str | None = None):
    """Encode query text, returning unpooled ColBERT token vectors as float32.

    Queries are short (~20-30 tokens) so no pooling needed — unpooled gives
    finer-grained matching against pooled document tokens.

    Returns: numpy array [n_tokens, dim] float32.
    """
    import numpy as np

    tokenizer, session = _get_model(model_path)
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    colbert_vectors = outputs[2]  # [1, n_tokens, 1024]
    return _apply_colbert_projection(colbert_vectors[0])


# ── Sparse format conversion ─────────────────────────────────────


def sparse_to_sparsevec(sparse: dict[int, float], dim: int = VOCAB_SIZE) -> str:
    """Convert sparse dict {token_id: weight} to pgvector sparsevec format.

    Format: '{idx1:val1, idx2:val2, ...}/dim'
    Indices are 1-based for pgvector.
    """
    if not sparse:
        return f"{{}}/{dim}"
    pairs = sorted(sparse.items())
    for tid, _ in pairs:
        if not (0 <= tid < dim):
            raise ValueError(f"Token ID {tid} out of bounds for vocab size {dim}")
    entries = ",".join(f"{tid + 1}:{weight:.6f}" for tid, weight in pairs)
    return f"{{{entries}}}/{dim}"
