from .additive_ntt import AdditiveNTT
from .binary_fields import BinaryField, BinaryFieldElement
from .tower_algebra import TowerAlgebra

from copy import deepcopy


class ReedSolomonCode:

    def __init__(self, log_dimension: int, log_inv_rate: int, field: BinaryField):
        assert isinstance(field, BinaryField)
        self.log_dimension = log_dimension
        self.log_inv_rate = log_inv_rate
        self.log_length = self.log_dimension + self.log_inv_rate
        self.field = field
        self.ntt = AdditiveNTT(self.log_dimension, self.log_length, self.field)

    def __repr__(self) -> str:
        return f"ReedSolomonCode(log_dimension={self.log_dimension}, log_inv_rate={self.log_inv_rate}) in {self.field}"

    def encode(
        self, data: list[BinaryFieldElement] | list[TowerAlgebra]
    ) -> list[BinaryFieldElement] | list[TowerAlgebra]:
        assert len(data) == 1 << self.log_dimension
        assert all(isinstance(v, BinaryFieldElement) for v in data) or all(
            isinstance(v, TowerAlgebra) for v in data
        )

        encoded = deepcopy(data)
        for _ in range(self.log_inv_rate):
            encoded += deepcopy(encoded)

        return self.ntt.forward_transform(encoded)


if __name__ == "__main__":

    from binary_fields import *
    from copy import deepcopy

    field = BF128
    log_dimension = 4
    log_inv_rate = 2
    rscode = ReedSolomonCode(log_dimension, log_inv_rate, field)

    data = [field.random_element() for _ in range(1 << log_dimension)]
    encoded = rscode.encode(data)
    print(encoded)
    print(len(encoded))
