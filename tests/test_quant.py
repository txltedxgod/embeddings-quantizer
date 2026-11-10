import numpy as np
import pytest
from fastapi.testclient import TestClient

from quantizer.api import app
from quantizer.pq import ProductQuantizer
from quantizer.sq8 import BinaryQuantizer, ScalarQuantizer8Bit


def test_sq8() -> None:
    q = ScalarQuantizer8Bit()
    v = [0.12, -0.45, 0.88, 0.05, -0.22, 0.99]
    res = q.quantize(v)
    assert res["reconstruction_mse"] < 0.01
    assert res["compressed_bytes_len"] == len(v)


def test_sq8_asymmetric_dot_search_matches_exact_ranking() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((40, 64)).astype(np.float32)
    query = vectors[12] + rng.normal(0.0, 0.05, size=64).astype(np.float32)

    q = ScalarQuantizer8Bit()
    codes, v_mins, scales = q.encode(vectors)
    hits = q.asymmetric_dot_search(query, codes, v_mins, scales, top_k=3)
    assert hits[0][0] == 12
    assert hits[0][1] > hits[1][1]

    with pytest.raises(ValueError, match="finite"):
        q.quantize([0.1, float("nan"), 0.2])


def test_binary_quantization_hamming_search() -> None:
    bq = BinaryQuantizer()
    vectors = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            [0.8, 0.5, 0.9, -0.2, -0.4, -0.9, -0.3, -0.1],
        ],
        dtype=np.float32,
    )
    packed = bq.encode(vectors)
    assert packed.shape == (3, 1)

    hits = bq.hamming_search([0.9, 0.7, 0.2, 0.4, -0.5, -0.8, -0.1, -0.2], packed, top_k=2)
    assert hits[0][0] == 0
    assert hits[0][1] == 0


def test_product_quantizer_fit_encode_and_adc_search() -> None:
    rng = np.random.default_rng(123)
    # Create structured cluster corpus so PQ reconstructs accurately and preserves nearest neighbor
    centers = rng.standard_normal((8, 32)).astype(np.float32) * 3.0
    corpus = np.vstack([c + rng.normal(0.0, 0.1, size=(12, 32)) for c in centers]).astype(np.float32)

    pq = ProductQuantizer(n_subspaces=4, n_centroids=16, n_iter=15, seed=42)
    pq.fit(corpus)
    codes = pq.encode(corpus)
    assert codes.shape == (96, 4)
    assert codes.dtype == np.uint8
    assert pq.compression_ratio() == 32.0

    recon = pq.decode(codes)
    mse = float(np.mean((corpus - recon) ** 2))
    assert mse < 0.25

    query = centers[3] + np.full(32, 0.02, dtype=np.float32)
    adc_hits = pq.asymmetric_search(query, codes, top_k=5)
    # Center 3 occupies corpus rows 36..47
    assert 36 <= adc_hits[0][0] < 48


def test_http_api_quantization_endpoints() -> None:
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    sq8_resp = client.post(
        "/api/v1/quantize",
        json={"vector": [0.1, 0.5, -0.3, 0.9], "include_codes": True},
    )
    assert sq8_resp.status_code == 200
    assert len(sq8_resp.json()["codes"]) == 4

    bin_resp = client.post(
        "/api/v1/quantize/binary",
        json={
            "vectors": [[1.0, 2.0, -1.0, -2.0, 1.0, 1.0, -1.0, -1.0], [-1.0] * 8],
            "query": [0.5, 0.9, -0.3, -0.2, 0.1, 0.4, -0.8, -0.5],
            "top_k": 1,
        },
    )
    assert bin_resp.status_code == 200
    assert bin_resp.json()["nearest"][0]["index"] == 0

    pq_resp = client.post(
        "/api/v1/pq/fit-encode",
        json={
            "vectors": [[float(i), float(i), 0.0, 0.0, 1.0, 1.0, -1.0, -1.0] for i in range(10)],
            "n_subspaces": 2,
            "n_centroids": 4,
            "query": [9.0, 9.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
            "top_k": 2,
        },
    )
    assert pq_resp.status_code == 200
    assert pq_resp.json()["nearest"][0]["index"] in (7, 8, 9)
