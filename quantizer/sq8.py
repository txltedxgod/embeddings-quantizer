import numpy as np
from typing import Dict, Any, List

class ScalarQuantizer8Bit:
    def quantize(self, vector: List[float]) -> Dict[str, Any]:
        arr = np.array(vector, dtype=np.float32)
        v_min, v_max = float(arr.min()), float(arr.max())
        scale = (v_max - v_min) / 255.0 if v_max != v_min else 1.0
        q_bytes = np.round((arr - v_min) / scale).astype(np.uint8)

        # Dequantize test
        recon = (q_bytes.astype(np.float32) * scale) + v_min
        mse = float(np.mean((arr - recon) ** 2))

        return {
            "original_dim": len(vector),
            "compressed_bytes_len": len(q_bytes),
            "compression_ratio": "4x (75% memory saved)",
            "reconstruction_mse": round(mse, 6)
        }
