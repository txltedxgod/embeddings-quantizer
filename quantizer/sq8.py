"""Scalar (SQ8) and 1-bit Binary (BQ) vector quantizers with vectorized asymmetric search."""

from __future__ import annotations

from typing import Any

import numpy as np

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _validate_vectors(arr: np.ndarray, ndim: int) -> np.ndarray:
    if arr.ndim != ndim or arr.size == 0:
        raise ValueError(f"Expected non-empty {ndim}D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input vectors must contain only finite floating-point numbers")
    return arr.astype(np.float32, copy=False)


class ScalarQuantizer8Bit:
    """Asymmetric 8-bit uint8 scalar quantizer achieving 4x vector compression."""

    def encode(
        self, vectors: list[list[float]] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a batch (N, D) of float32 vectors into (uint8 codes, v_mins, scales)."""
        arr = _validate_vectors(np.asarray(vectors), ndim=2)
        v_mins = arr.min(axis=1)
        v_maxs = arr.max(axis=1)
        spans = v_maxs - v_mins
        scales = np.where(spans > 0.0, spans / np.float32(255.0), np.float32(1.0))

        normalized = (arr - v_mins[:, None]) / scales[:, None]
        codes = np.clip(np.round(normalized), 0, 255).astype(np.uint8)
        return codes, v_mins.astype(np.float32), scales.astype(np.float32)

    def decode(
        self,
        codes: np.ndarray,
        v_mins: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct float32 approximations from uint8 codes and per-vector calibration."""
        return (codes.astype(np.float32) * scales[:, None]) + v_mins[:, None]

    def asymmetric_dot_search(
        self,
        query: list[float] | np.ndarray,
        codes: np.ndarray,
        v_mins: np.ndarray,
        scales: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Compute inner-product scores directly over uint8 codes without full float32 expansion."""
        q = _validate_vectors(np.asarray(query), ndim=1)
        if q.shape[0] != codes.shape[1]:
            raise ValueError(f"Query dim {q.shape[0]} does not match codes dim {codes.shape[1]}")
        if top_k <= 0 or codes.shape[0] == 0:
            return []

        raw_dots = codes.astype(np.float32) @ q
        scores = raw_dots * scales + float(q.sum()) * v_mins
        k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:k]
        return [(int(idx), round(float(scores[idx]), 6)) for idx in top_idx]

    def quantize(
        self,
        vector: list[float],
        include_codes: bool = False,
    ) -> dict[str, Any]:
        """Quantize a single vector and report compression & reconstruction MSE metrics."""
        arr = _validate_vectors(np.asarray(vector), ndim=1)
        codes, v_mins, scales = self.encode(arr[None, :])
        recon = self.decode(codes, v_mins, scales)[0]
        mse = float(np.mean((arr - recon) ** 2))

        result: dict[str, Any] = {
            "original_dim": len(vector),
            "compressed_bytes_len": int(codes.shape[1]),
            "compression_ratio": "4x (75% memory saved)",
            "reconstruction_mse": round(mse, 6),
        }
        if include_codes:
            result["codes"] = codes[0].tolist()
            result["v_min"] = round(float(v_mins[0]), 6)
            result["scale"] = round(float(scales[0]), 8)
        return result


class BinaryQuantizer:
    """1-bit sign quantizer packing 32 floats per uint32 equivalent (32x compression)."""

    def encode(self, vectors: list[list[float]] | np.ndarray) -> np.ndarray:
        """Threshold vectors around 0 and pack bits into uint8 byte arrays."""
        arr = _validate_vectors(np.asarray(vectors), ndim=2)
        bits = (arr >= 0.0).astype(np.uint8)
        return np.packbits(bits, axis=1, bitorder="big")

    def hamming_search(
        self,
        query: list[float] | np.ndarray,
        packed_codes: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, int]]:
        """Rank packed binary codes by ascending Hamming distance to the thresholded query."""
        q = _validate_vectors(np.asarray(query), ndim=1)
        if top_k <= 0 or packed_codes.shape[0] == 0:
            return []
        q_packed = self.encode(q[None, :])[0]
        if q_packed.shape[0] != packed_codes.shape[1]:
            raise ValueError("Packed query length does not match code width")

        xor_bytes = np.bitwise_xor(packed_codes, q_packed[None, :])
        distances = _POPCOUNT_LUT[xor_bytes].sum(axis=1)
        k = min(top_k, len(distances))
        top_idx = np.argsort(distances)[:k]
        return [(int(idx), int(distances[idx])) for idx in top_idx]
