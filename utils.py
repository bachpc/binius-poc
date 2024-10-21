def tensor_product(coordinates, field):
    o = [field.ONE]
    for coord in coordinates:
        p0 = [(field.ONE - coord) * v for v in o]
        p1 = [c0 + v for c0, v in zip(p0, o)]
        o = p0 + p1
    return o


def inner_product(xs, ys, field):
    assert len(xs) == len(ys)
    return sum((x * y for x, y in zip(xs, ys)), field.ZERO)


def matrix_multiply_vector(mat, vec, field):
    return [inner_product(row, vec, field) for row in mat]


def vector_multiply_matrix(vec, mat, field):
    return matrix_multiply_vector(transpose(mat), vec, field)


def log2(x):
    assert x & (x - 1) == 0
    return x.bit_length() - 1


def transpose(mat):
    nrows, ncols = len(mat), len(mat[0])
    return [[mat[i][j] for i in range(nrows)] for j in range(ncols)]
