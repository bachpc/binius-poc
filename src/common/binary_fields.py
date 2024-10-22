from collections.abc import Iterable
import random


class BinaryField:

    def __init__(self, bit_length: int):
        # assert bit_length > 0 and bit_length & (bit_length - 1) == 0
        assert bit_length in {1, 2, 4, 8, 16, 32, 64, 128}
        self.bit_length = bit_length
        self.ZERO = self(0)
        self.ONE = self(1)

    def __call__(self, value: "int | BinaryFieldElement") -> "BinaryFieldElement":
        return BinaryFieldElement(self, value)

    def __hash__(self) -> int:
        return hash(("BinaryField", self.bit_length))

    def __eq__(self, other: "BinaryField") -> bool:
        assert isinstance(other, BinaryField)
        return self.bit_length == other.bit_length

    def __repr__(self) -> str:
        return f"Binary Field of Size 2^{self.bit_length}"

    def to_bytes(self) -> bytes:
        return self.bit_length.to_bytes(2, "little")

    def is_extension_of(self, other: "BinaryField") -> bool:
        assert isinstance(other, BinaryField)
        return self.bit_length % other.bit_length == 0

    def random_element(self) -> "BinaryFieldElement":
        value = random.getrandbits(self.bit_length)
        return self(value)

    def order(self) -> int:
        return (1 << self.bit_length) - 1

    def degree(self, subfield: "BinaryField"):
        assert isinstance(subfield, BinaryField)
        assert self.is_extension_of(subfield)
        return self.bit_length // subfield.bit_length

    def check_element(self, element: "BinaryFieldElement") -> bool:
        return isinstance(element, BinaryFieldElement) and element.field == self

    def from_unpacked(
        self, elts: Iterable["BinaryFieldElement"]
    ) -> "BinaryFieldElement":
        assert all(isinstance(e, BinaryFieldElement) for e in elts)
        assert sum(e.bit_length for e in elts) == self.bit_length
        ret = 0
        for e in elts:
            ret = ret << e.bit_length | e.value
        return BinaryFieldElement(self, ret)

    def cast_slice(
        self, elts: Iterable["BinaryFieldElement"]
    ) -> list["BinaryFieldElement"]:
        assert all(isinstance(e, BinaryFieldElement) for e in elts)
        assert sum(e.bit_length for e in elts) % self.bit_length == 0
        x = 0
        for e in elts:
            x = x << e.bit_length | e.value
        ret = []
        mask = (1 << self.bit_length) - 1
        width = sum(e.bit_length for e in elts) // self.bit_length
        for _ in range(width):
            ret.append(BinaryFieldElement(self, x & mask))
            x >>= self.bit_length
        return ret[::-1]

    def __or__(self, other: "BinaryField") -> "BinaryField":
        assert isinstance(other, BinaryField)
        if self.is_extension_of(other):
            return self
        return other

    def __and__(self, other: "BinaryField") -> "BinaryField":
        assert isinstance(other, BinaryField)
        if self.is_extension_of(other):
            return other
        return self


class BinaryFieldElement:

    def __init__(self, field: "BinaryField", value: "int | BinaryFieldElement"):
        assert isinstance(field, BinaryField)
        if isinstance(value, BinaryFieldElement):
            value = value.value
        assert value >= 0 and value.bit_length() <= field.bit_length
        self.field = field
        self.value = value

    def __hash__(self) -> int:
        return hash(("BinaryFieldElement", self.field, self.value))

    def __eq__(self, other: "BinaryFieldElement") -> bool:
        assert isinstance(other, BinaryFieldElement)
        return self.field == other.field and self.value == other.value

    def __add__(self, other: "BinaryFieldElement") -> "BinaryFieldElement":
        assert isinstance(other, BinaryFieldElement)
        field = self.field if self.field.is_extension_of(other.field) else other.field
        value = self.value ^ other.value
        return BinaryFieldElement(field, value)

    __sub__ = __add__

    def __neg__(self) -> "BinaryFieldElement":
        return self

    def __mul__(self, other: "BinaryFieldElement") -> "BinaryFieldElement":

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
            return L1L2 ^ R1R2 ^ ((Z3 ^ L1L2 ^ R1R2 ^ R1R2_high) << halflen)

        assert isinstance(other, BinaryFieldElement)
        if self.field.is_extension_of(other.field):
            unpacked = self.unpack_into(other.field)
            return self.field.from_unpacked(
                [
                    BinaryFieldElement(
                        other.field,
                        mul_equal_length(v.value, other.value, other.bit_length),
                    )
                    for v in unpacked
                ]
            )
        else:
            unpacked = other.unpack_into(self.field)
            return other.field.from_unpacked(
                [
                    BinaryFieldElement(
                        self.field,
                        mul_equal_length(v.value, self.value, self.bit_length),
                    )
                    for v in unpacked
                ]
            )

    def __pow__(self, n: int) -> "BinaryFieldElement":
        result = self.field.ONE
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            base = base * base
        return result

    def inv(self) -> "BinaryFieldElement":
        return self ** (self.field.order() - 1)

    def __truediv__(self, other: "BinaryFieldElement") -> "BinaryFieldElement":
        assert isinstance(other, BinaryFieldElement)
        return self * other.inv()

    def __repr__(self):
        return f"{self.value:#0{(self.bit_length >> 2) + 2}x}"

    def to_bytes(self) -> bytes:
        return self.field.to_bytes() + self.value.to_bytes(
            (self.bit_length + 7) // 8, "little"
        )

    @property
    def bit_length(self) -> int:
        return self.field.bit_length

    def copy(self) -> "BinaryFieldElement":
        return BinaryFieldElement(self.field, self.value)

    def to_extension_field(self, ext_field: "BinaryField") -> "BinaryFieldElement":
        assert isinstance(ext_field, BinaryField)
        assert ext_field.is_extension_of(self.field)
        return BinaryFieldElement(ext_field, self.value)

    def unpack_into(self, subfield: "BinaryField") -> tuple["BinaryFieldElement", ...]:
        assert isinstance(subfield, BinaryField)
        assert self.field.is_extension_of(subfield)
        width = self.field.degree(subfield)
        mask = (1 << subfield.bit_length) - 1
        ret = []
        x = self.value
        for _ in range(width):
            ret.append(BinaryFieldElement(subfield, x & mask))
            x >>= subfield.bit_length
        return tuple(ret[::-1])


BF1 = BinaryField(1)
BF2 = BinaryField(2)
BF4 = BinaryField(4)
BF8 = BinaryField(8)
BF16 = BinaryField(16)
BF32 = BinaryField(32)
BF64 = BinaryField(64)
BF128 = BinaryField(128)


if __name__ == "__main__":
    a, b, c = [BF128.random_element() for _ in range(3)]
    b = BF16.random_element()
    print(a)
    print(b)
    print(c)
    print((a + b) * c == a * c + b * c)
    print((a / b) * b == a)
    print(BF128)
    print(BF128.order())
    print(a.unpack_into(BF32))
    print(a == BinaryFieldElement.pack_from(a.unpack_into(BF16), BF128))
    print(BF128 == BinaryField(128))
