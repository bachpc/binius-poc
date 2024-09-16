# eval multilinear polynomial at point
# `evals` are the evaluations of multilinear polynomial over hypercube {0, 1}^k
def multilinear_poly_eval(evals, point, field):
    assert len(evals) == 2 ** len(point)
    return _multilinear_poly_eval(evals, point, field)


def _multilinear_poly_eval(evals, point, field):
    if len(point) == 0:
        return evals[0]
    top = _multilinear_poly_eval(evals[:len(evals) // 2], point[:-1], field)
    bottom = _multilinear_poly_eval(evals[len(evals) // 2:], point[:-1], field)
    return (
        (bottom - top) * point[-1] + top
    )


def evaluation_tensor_product(point, field):
    o = [field.ONE]
    for coord in point:
        p0 = [(field.ONE - coord) * v for v in o]
        p1 = [c0 + v for c0, v in zip(p0, o)]
        o = p0 + p1
    return o


def log2(x):
    assert x & (x - 1) == 0
    return x.bit_length() - 1


def transpose(mat):
    nrows, ncols = len(mat), len(mat[0])
    return [
        [mat[i][j] for i in range(nrows)]
        for j in range(ncols)
    ]
