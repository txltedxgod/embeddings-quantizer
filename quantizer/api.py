from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from quantizer.sq8 import ScalarQuantizer8Bit

app = FastAPI(title="Embeddings Quantizer", version="0.1.0")
sq8 = ScalarQuantizer8Bit()

class QuantReq(BaseModel):
    vector: List[float]

@app.post("/api/v1/quantize")
def quantize(req: QuantReq):
    return sq8.quantize(req.vector)
