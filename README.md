# Embeddings Quantizer

[![CI](https://github.com/txltedxgod/embeddings-quantizer/actions/workflows/ci.yml/badge.svg)](https://github.com/txltedxgod/embeddings-quantizer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

High-performance vector quantization engine in Python 3.11+ & NumPy implementing 8-bit Scalar Quantization (SQ8), 1-bit Binary Quantization (BQ), and Product Quantization (PQ) with K-Means codebooks and Asymmetric Distance Computation (ADC).

## Features
- **SQ8 (`ScalarQuantizer8Bit`)**: 4x compression with per-vector calibration and direct uint8 asymmetric inner-product ranking (`asymmetric_dot_search`).
- **1-Bit Binary (`BinaryQuantizer`)**: 32x packed bit compression with fast lookup-table Hamming ranking.
- **Product Quantization (`ProductQuantizer`)**: Orthogonal subspace K-Means training (`fit`/`encode`/`decode`) and ADC LUT nearest-neighbor search (`asymmetric_search`).
- **REST API**: `/health`, `POST /api/v1/quantize`, `POST /api/v1/quantize/binary`, `POST /api/v1/pq/fit-encode`.

## Testing
```bash
ruff check .
pytest -v tests/
```
