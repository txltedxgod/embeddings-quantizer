from quantizer.sq8 import ScalarQuantizer8Bit

def test_sq8():
    q = ScalarQuantizer8Bit()
    v = [0.12, -0.45, 0.88, 0.05, -0.22, 0.99]
    res = q.quantize(v)
    assert res["reconstruction_mse"] < 0.01
