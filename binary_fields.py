import random


class BinaryField:

    def __init__(self, bit_length, multiplicative_generator):
        # assert bit_length > 0 and bit_length & (bit_length - 1) == 0
        assert bit_length in {1, 2, 4, 8, 16, 32, 64, 128}
        self.bit_length = bit_length
        self.MULTIPLICATIVE_GENERATOR = self(multiplicative_generator)
        self.ZERO = self(0)
        self.ONE = self(1)

    def __call__(self, val):
        return BinaryFieldElement(self, val)

    def __hash__(self):
        return hash(hash(self.bit_length) + 0x1337 * hash(self.MULTIPLICATIVE_GENERATOR.value))

    def __repr__(self):
        return f'Binary Field of Size 2^{self.bit_length} with generator {self.MULTIPLICATIVE_GENERATOR}'

    def is_extension_of(self, other):
        assert isinstance(other, BinaryField)
        return self.bit_length % other.bit_length == 0

    def random_element(self):
        value = random.getrandbits(self.bit_length)
        return self(value)

    def order(self):
        return (1 << self.bit_length) - 1

    def pack_width(self, subfield):
        assert isinstance(subfield, BinaryField)
        assert self.is_extension_of(subfield)
        return self.bit_length // subfield.bit_length


class BinaryFieldElement:

    def __init__(self, field, value):
        assert isinstance(field, BinaryField)
        if isinstance(value, BinaryFieldElement):
            value = value.value
        assert value >= 0 and value.bit_length() <= field.bit_length
        self.field = field
        self.value = value

    def __hash__(self):
        return hash(hash(self.field) + 0x1339 * hash(self.value))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__() and self.value == other.value

    def __add__(self, other):
        assert isinstance(other, BinaryFieldElement)
        field = self.field if self.field.is_extension_of(other.field) else other.field
        value = self.value ^ other.value
        return BinaryFieldElement(field, value)

    __sub__ = __add__

    def __neg__(self):
        return self

    def __mul__(self, other):

        # multiply using Karatsuba method
        def mul_equal_length(v1, v2, length):
            if v1 < 2 or v2 < 2:
                return v1 * v2
            halflen = length >> 1
            quarterlen = length >> 2
            halfmask = (1 << halflen) - 1

            L1, R1 = v1 & halfmask, v1 >> halflen
            L2, R2 = v2 & halfmask, v2 >> halflen

            if (L1, R1) == (0, 1):
                outR = mul_equal_length(1 << quarterlen, R2, halflen) ^ L2
                return R2 ^ (outR << halflen)

            L1L2 = mul_equal_length(L1, L2, halflen)
            R1R2 = mul_equal_length(R1, R2, halflen)
            R1R2_high = mul_equal_length(1 << quarterlen, R1R2, halflen)
            Z3 = mul_equal_length(L1 ^ R1, L2 ^ R2, halflen)
            return (
                L1L2 ^
                R1R2 ^
                ((Z3 ^ L1L2 ^ R1R2 ^ R1R2_high) << halflen)
            )

        assert isinstance(other, BinaryFieldElement)
        if self.field.is_extension_of(other.field):
            unpacked = self.unpack_into(other.field)
            return BinaryFieldElement.pack_from(
                [
                    BinaryFieldElement(other.field, mul_equal_length(v.value, other.value, other.bit_length))
                    for v in unpacked
                ],
                self.field
            )
        else:
            unpacked = other.unpack_into(self.field)
            return BinaryFieldElement.pack_from(
                [
                    BinaryFieldElement(self.field, mul_equal_length(v.value, self.value, self.bit_length))
                    for v in unpacked
                ],
                other.field
            )

    def __pow__(self, n):
        result = self.field.ONE
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            base = base * base
        return result

    def inv(self):
        return self ** (self.field.order() - 1)

    def __truediv__(self, other):
        assert isinstance(other, BinaryFieldElement)
        return self * other.inv()

    def __repr__(self):
        return f'{self.value:#0{(self.bit_length >> 2) + 2}x}'

    def to_bytes(self, length, byteorder):
        assert length >= (self.bit_length + 7) // 8
        return self.value.to_bytes(length, byteorder)

    @property
    def bit_length(self):
        return self.field.bit_length

    def copy(self):
        return BinaryFieldElement(self.field, self.value)

    def unpack_into(self, subfield):
        assert isinstance(subfield, BinaryField)
        assert self.field.is_extension_of(subfield)
        width = self.field.pack_width(subfield)
        unpacked = [
            (self.value >> (self.bit_length - subfield.bit_length * (i + 1))) & ((1 << subfield.bit_length) - 1)
            for i in range(0, width)
        ]
        return tuple(BinaryFieldElement(subfield, v) for v in unpacked)

    @staticmethod
    def pack_from(elts, field):
        assert isinstance(field, BinaryField)
        assert all(isinstance(e, BinaryFieldElement) for e in elts)
        assert len(set(e.field for e in elts)) == 1
        assert elts[0].bit_length * len(elts) == field.bit_length
        subfield_length = elts[0].bit_length
        return BinaryFieldElement(
            field,
            sum(e.value << (field.bit_length - subfield_length * (i + 1)) for i, e in enumerate(elts))
        )


BF1 = BinaryField(1, 0x1)
BF2 = BinaryField(2, 0x2)
BF4 = BinaryField(4, 0x5)
BF8 = BinaryField(8, 0x2D)
BF16 = BinaryField(16, 0xE2DE)
BF32 = BinaryField(32, 0x03E21CEA)
BF64 = BinaryField(64, 0x070F870DCD9C1D88)
BF128 = BinaryField(128, 0x2E895399AF449ACE499596F6E5FCCAFA)


if __name__ == '__main__':
    a, b, c = [BF128.random_element() for _ in range(3)]
    b = BF16.random_element()
    print(a)
    print(b)
    print(c)
    print((a + b) * c)
    print(a * c + b * c)
    print((a / b) * b)
    print(BF128)
    print(BF128.MULTIPLICATIVE_GENERATOR ** (BF128.order() // 17))
    print(a ** (BF128.order() // 17))
    print(a)
    print(BF128.order())
    print(a * BF128(2))
