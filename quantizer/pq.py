"""Product Quantization (PQ) engine with K-Means subspace codebooks and ADC search."""

from __future__ import annotations

import numpy as np

from quantizer.sq8 import _validate_vectors


def _kmeans_subspace(
    data: np.ndarray,
    n_centroids: int,
    n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Train K centroids on a single subspace slice (N, d_sub) using Lloyd's algorithm."""
    n_samples = data.shape[0]
    if n_samples <= n_centroids:
        pad = np.repeat(data[-1:], n_centroids - n_samples, axis=0)
        return np.vstack([data, pad]).astype(np.float32)

    init_idx = rng.choice(n_samples, size=n_centroids, replace=False)
    centroids = data[init_idx].copy()

    for _ in range(n_iter):
        # Compute squared Euclidean distances between samples and centroids
        dists = (
            np.sum(data * data, axis=1, keepdims=True)
            + np.sum(centroids * centroids, axis=1)[None, :]
            - 2.0 * (data @ centroids.T)
        )
        labels = np.argmin(dists, axis=1)

        updated = centroids.copy()
        for k in range(n_centroids):
            mask = labels == k
            if np.any(mask):
                updated[k] = data[mask].mean(axis=0)
            else:
                updated[k] = data[rng.integers(0, n_samples)]
        if np.allclose(updated, centroids, atol=1e-5):
            centroids = updated
            break
        centroids = updated

    return centroids.astype(np.float32)


class ProductQuantizer:
    """Orthogonal subspace Product Quantizer supporting uint8 codes and ADC search."""

    def __init__(
        self,
        n_subspaces: int = 4,
        n_centroids: int = 16,
        n_iter: int = 15,
        seed: int = 42,
    ) -> None:
        if n_subspaces < 1:
            raise ValueError("n_subspaces must be at least 1")
        if not 1 <= n_centroids <= 256:
            raise ValueError("n_centroids must be in [1, 256] to fit in uint8 codes")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")

        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        self.seed = seed
        self.dim: int | None = None
        self.sub_dim: int | None = None
        self.codebooks: np.ndarray | None = None

    def fit(self, corpus: list[list[float]] | np.ndarray) -> ProductQuantizer:
        """Fit M subspace K-Means codebooks on training vectors of shape (N, D)."""
        arr = _validate_vectors(np.asarray(corpus), ndim=2)
        dim = arr.shape[1]
        if dim % self.n_subspaces != 0:
            raise ValueError(f"Vector dim {dim} must be divisible by n_subspaces {self.n_subspaces}")

        sub_dim = dim // self.n_subspaces
        rng = np.random.default_rng(self.seed)
        codebooks = np.zeros((self.n_subspaces, self.n_centroids, sub_dim), dtype=np.float32)

        for m in range(self.n_subspaces):
            sub_slice = arr[:, m * sub_dim : (m + 1) * sub_dim]
            codebooks[m] = _kmeans_subspace(sub_slice, self.n_centroids, self.n_iter, rng)

        self.dim = dim
        self.sub_dim = sub_dim
        self.codebooks = codebooks
        return self

    def _require_fitted(self, dim: int | None = None) -> tuple[int, int, np.ndarray]:
        if self.codebooks is None or self.dim is None or self.sub_dim is None:
            raise RuntimeError("ProductQuantizer must be fitted before calling encode/decode/search")
        if dim is not None and dim != self.dim:
            raise ValueError(f"Expected vector dim {self.dim}, got {dim}")
        return self.dim, self.sub_dim, self.codebooks

    def encode(self, vectors: list[list[float]] | np.ndarray) -> np.ndarray:
        """Encode vectors of shape (N, D) into compact uint8 centroid indices (N, M)."""
        arr = _validate_vectors(np.asarray(vectors), ndim=2)
        _, sub_dim, codebooks = self._require_fitted(arr.shape[1])

        codes = np.empty((arr.shape[0], self.n_subspaces), dtype=np.uint8)
        for m in range(self.n_subspaces):
            sub_vec = arr[:, m * sub_dim : (m + 1) * sub_dim]
            centroids = codebooks[m]
            dists = (
                np.sum(sub_vec * sub_vec, axis=1, keepdims=True)
                + np.sum(centroids * centroids, axis=1)[None, :]
                - 2.0 * (sub_vec @ centroids.T)
            )
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate float32 vectors of shape (N, D) from PQ codes (N, M)."""
        dim, sub_dim, codebooks = self._require_fitted()
        if codes.ndim != 2 or codes.shape[1] != self.n_subspaces:
            raise ValueError(f"Expected codes shape (N, {self.n_subspaces}), got {codes.shape}")

        out = np.empty((codes.shape[0], dim), dtype=np.float32)
        for m in range(self.n_subspaces):
            out[:, m * sub_dim : (m + 1) * sub_dim] = codebooks[m][codes[:, m]]
        return out

    def asymmetric_search(
        self,
        query: list[float] | np.ndarray,
        codes: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Search PQ codes using Asymmetric Distance Computation (ADC) lookup tables."""
        q = _validate_vectors(np.asarray(query), ndim=1)
        _, sub_dim, codebooks = self._require_fitted(q.shape[0])
        if codes.ndim != 2 or codes.shape[1] != self.n_subspaces:
            raise ValueError(f"Expected codes shape (N, {self.n_subspaces}), got {codes.shape}")
        if top_k <= 0 or codes.shape[0] == 0:
            return []

        # Build distance LUT of shape (M, K) for the exact unquantized query sub-vectors
        lut = np.empty((self.n_subspaces, self.n_centroids), dtype=np.float32)
        for m in range(self.n_subspaces):
            q_sub = q[m * sub_dim : (m + 1) * sub_dim]
            diff = codebooks[m] - q_sub[None, :]
            lut[m] = np.sum(diff * diff, axis=1)

        m_idx = np.arange(self.n_subspaces)[None, :]
        approx_l2_sq = lut[m_idx, codes].sum(axis=1)

        k = min(top_k, len(approx_l2_sq))
        best = np.argsort(approx_l2_sq)[:k]
        return [(int(idx), round(float(approx_l2_sq[idx]), 6)) for idx in best]

    def compression_ratio(self) -> float:
        """Compute theoretical float32-to-uint8 compression factor."""
        dim, _, _ = self._require_fitted()
        return (dim * 4.0) / float(self.n_subspaces)
