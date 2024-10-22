from .binary_fields import BinaryField, BinaryFieldElement
from .tower_algebra import TowerAlgebra

from collections.abc import Iterable
from hashlib import sha256


class Challenger:

    def __init__(self, seed: bytes = b"init_challenger"):
        self.state = self._hash(seed)
        self.counter = 0

    def _hash(self, x: bytes):
        return sha256(x).digest()

    def observe(self, value: bytes | BinaryFieldElement | TowerAlgebra):
        if isinstance(value, BinaryFieldElement):
            self.state = self._hash(self.state + b"1" + value.to_bytes())
        elif isinstance(value, bytes):
            self.state = self._hash(self.state + b"2" + value)
        elif isinstance(value, TowerAlgebra):
            self.observe_slice(value.elems[:])
        else:
            assert False
        self.counter = 0

    def observe_slice(self, values: Iterable[BinaryFieldElement]):
        for value in values:
            self.observe(value)

    def _sample(self) -> int:
        value = int.from_bytes(
            self._hash(self.state + b"@" + self.counter.to_bytes(8, "little"))
        )
        self.counter += 1
        return value

    def sample(self, field: BinaryField) -> BinaryFieldElement:
        assert isinstance(field, BinaryField)
        value = self._sample() & ((1 << field.bit_length) - 1)
        return BinaryFieldElement(field, value)

    def sample_vec(self, n: int, field: BinaryField) -> list[BinaryFieldElement]:
        return [self.sample(field) for _ in range(n)]

    def sample_bits(self, bits: int) -> int:
        return self._sample() & ((1 << bits) - 1)
