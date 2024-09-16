from utils import log2
from functools import lru_cache


@lru_cache
def s(k, x):
    if k == 0:
        return x
    tmp = s(k - 1, x)
    return tmp * tmp + tmp


# Additive Fast Fourier Transform
# see https://arxiv.org/pdf/1802.03932
def fft(coeffs, k, a, field=None):
    if k == -1:
        return coeffs[:]
    if not field:
        field = a.field
    assert len(coeffs) == 1 << (k + 1)
    w = s(k, a)

    p0 = coeffs[:len(coeffs) // 2]
    p1 = coeffs[len(coeffs) // 2:]
    h0 = [c0 + c1 * w for c0, c1 in zip(p0, p1)]
    h1 = [c0 + c1 for c0, c1 in zip(h0, p1)]

    return fft(h0, k - 1, a, field) + fft(h1, k - 1, a + field(1 << k), field)


def ifft(points, k, a, field=None):
    if k == -1:
        return points[:]
    if not field:
        field = a.field
    w = s(k, a)

    h0 = ifft(points[:len(points) // 2], k - 1, a, field)
    h1 = ifft(points[len(points) // 2:], k - 1, a + field(1 << k), field)
    p1 = [c0 + c1 for c0, c1 in zip(h0, h1)]
    p0 = [c0 + c1 * w for c0, c1 in zip(h0, p1)]

    return p0 + p1


def reed_solomon_encode(field, data, inv_rate=2):
    k = log2(len(data)) - 1
    a = field.ONE

    result = []
    for _ in range(inv_rate):
        result += fft(data, k, a, field)
        a = a * field.MULTIPLICATIVE_GENERATOR

    return result


if __name__ == '__main__':

    from binary_fields import BF1, BF2, BF4, BF8, BF16, BF32, BF64, BF128

    field = BF128
    k = 3
    a = field.ONE

    coeffs = [field.random_element() for _ in range(1 << (k + 1))]
    print(coeffs)
    points = fft(coeffs, k, a, field)
    print(points)
    print(ifft(points, k, a, field))

    print(reed_solomon_encode(field, coeffs))
