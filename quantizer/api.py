"""FastAPI REST service exposing SQ8, Binary, and Product Quantization endpoints."""

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from quantizer.pq import ProductQuantizer
from quantizer.sq8 import BinaryQuantizer, ScalarQuantizer8Bit

app = FastAPI(title="Embeddings Quantizer", version="1.0.0")
sq8 = ScalarQuantizer8Bit()
binary_q = BinaryQuantizer()


class QuantReq(BaseModel):
    """Single-vector SQ8 quantization request."""

    vector: list[float] = Field(..., min_length=1)
    include_codes: bool = Field(default=False)


class BinaryQuantReq(BaseModel):
    """Batch 1-bit binary quantization request."""

    vectors: list[list[float]] = Field(..., min_length=1)
    query: list[float] | None = Field(default=None)
    top_k: int = Field(default=3, ge=1, le=50)


class ProductQuantReq(BaseModel):
    """Batch Product Quantization fit-encode-search request."""

    vectors: list[list[float]] = Field(..., min_length=1)
    n_subspaces: int = Field(default=4, ge=1, le=64)
    n_centroids: int = Field(default=8, ge=1, le=256)
    query: list[float] | None = Field(default=None)
    top_k: int = Field(default=3, ge=1, le=50)


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/quantize", tags=["SQ8"])
def quantize(req: QuantReq) -> dict[str, Any]:
    """Quantize a vector using 8-bit scalar quantization."""
    try:
        return sq8.quantize(req.vector, include_codes=req.include_codes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/quantize/binary", tags=["Binary"])
def quantize_binary(req: BinaryQuantReq) -> dict[str, Any]:
    """Pack vectors into 1-bit binary codes and optionally rank via Hamming distance."""
    try:
        packed = binary_q.encode(req.vectors)
        dim = len(req.vectors[0])
        response: dict[str, Any] = {
            "count": int(packed.shape[0]),
            "bytes_per_vector": int(packed.shape[1]),
            "compression_ratio": f"{round((dim * 4) / max(1, int(packed.shape[1])), 2)}x",
        }
        if req.query is not None:
            response["nearest"] = [
                {"index": idx, "hamming_distance": dist}
                for idx, dist in binary_q.hamming_search(req.query, packed, top_k=req.top_k)
            ]
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/pq/fit-encode", tags=["Product Quantization"])
def pq_fit_encode(req: ProductQuantReq) -> dict[str, Any]:
    """Train PQ codebooks, encode batch vectors, report MSE, and run optional ADC search."""
    try:
        pq = ProductQuantizer(n_subspaces=req.n_subspaces, n_centroids=req.n_centroids)
        arr = np.asarray(req.vectors, dtype=np.float32)
        pq.fit(arr)
        codes = pq.encode(arr)
        recon = pq.decode(codes)
        mse = float(np.mean((arr - recon) ** 2))
        response: dict[str, Any] = {
            "count": int(codes.shape[0]),
            "n_subspaces": req.n_subspaces,
            "n_centroids": req.n_centroids,
            "compression_ratio": f"{round(pq.compression_ratio(), 2)}x",
            "reconstruction_mse": round(mse, 6),
        }
        if req.query is not None:
            response["nearest"] = [
                {"index": idx, "approx_l2_sq": dist}
                for idx, dist in pq.asymmetric_search(req.query, codes, top_k=req.top_k)
            ]
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
