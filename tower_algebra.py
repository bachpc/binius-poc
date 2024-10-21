from binary_fields import BinaryField, BinaryFieldElement
from utils import transpose


class TowerAlgebra:

    def __init__(
        self,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
        elems: list[BinaryFieldElement],
    ):
        TowerAlgebra.check_fields(F, F_vertical, F_horizontal)

        self.F = F
        self.F_vertical = F_vertical
        self.F_horizontal = F_horizontal
        self.n_cols = F_vertical.degree(F)
        self.n_rows = F_horizontal.degree(F)

        assert len(elems) <= self.n_rows
        assert all(F_vertical.check_element(e) for e in elems)
        if len(elems) < self.n_rows:
            elems.extend([F_vertical(0) for _ in range(self.n_rows - len(elems))])
        self.elems = elems

    def __hash__(self) -> int:
        return hash(
            ("TowerAlgebra", self.F, self.F_vertical, self.F_horizontal, self.elems)
        )

    def __eq__(self, other: "TowerAlgebra") -> bool:
        assert isinstance(other, TowerAlgebra)
        return (
            self.F == other.F
            and self.F_vertical == other.F_vertical
            and self.F_horizontal == other.F_horizontal
            and self.elems == other.elems
        )

    def __repr__(self) -> str:
        return f"TowerAlgebra({self.elems})"

    @staticmethod
    def check_fields(
        F: BinaryField, F_vertical: BinaryField, F_horizontal: BinaryField
    ) -> bool:
        ret = (
            isinstance(F, BinaryField)
            and isinstance(F_vertical, BinaryField)
            and isinstance(F_horizontal, BinaryField)
        )
        ret &= F_vertical.is_extension_of(F) and F_horizontal.is_extension_of(F)
        return ret

    @classmethod
    def new(
        cls,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
        elems: list[BinaryFieldElement],
    ) -> "TowerAlgebra":
        return cls(F, F_vertical, F_horizontal, elems)

    @classmethod
    def zero(
        cls,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
    ) -> "TowerAlgebra":
        return cls(F, F_vertical, F_horizontal, [])

    @classmethod
    def from_tensor(
        cls,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
        vertical: BinaryFieldElement,
        horizontal: BinaryFieldElement,
    ) -> "TowerAlgebra":
        assert TowerAlgebra.check_fields(F, F_vertical, F_horizontal)
        assert F_vertical.check_element(vertical)
        assert F_horizontal.check_element(horizontal)

        elems = [base * vertical for base in horizontal.unpack_into(F)]
        return cls(F, F_vertical, F_horizontal, elems)

    @classmethod
    def from_vertical(
        cls,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
        vertical: BinaryFieldElement,
    ) -> "TowerAlgebra":
        assert TowerAlgebra.check_fields(F, F_vertical, F_horizontal)
        assert F_vertical.check_element(vertical)

        return cls(F, F_vertical, F_horizontal, [vertical])

    @classmethod
    def from_horizontal(
        cls,
        F: BinaryField,
        F_vertical: BinaryField,
        F_horizontal: BinaryField,
        horizontal: BinaryFieldElement,
    ) -> "TowerAlgebra":
        assert TowerAlgebra.check_fields(F, F_vertical, F_horizontal)
        assert F_horizontal.check_element(horizontal)

        elems = [
            base.to_extension_field(F_vertical) for base in horizontal.unpack_into(F)
        ]
        return cls(F, F_vertical, F_horizontal, elems)

    def try_extract_vertical(self) -> BinaryFieldElement:
        assert all(e == self.F_vertical.ZERO for e in self.elems[1:])
        return self.elems[0]

    def scale_vertical(self, scalar: BinaryFieldElement):
        assert self.F_vertical.check_element(scalar)
        elems = [scalar * e for e in self.elems]
        return TowerAlgebra(self.F, self.F_vertical, self.F_horizontal, elems)

    def scale_horizontal(self, scalar: BinaryFieldElement):
        assert self.F_horizontal.check_element(scalar)
        return self.transpose().scale_vertical(scalar).transpose()

    def transpose(self) -> "TowerAlgebra":
        mat = [e.unpack_into(self.F) for e in self.elems]
        horizontal_elems = [
            self.F_horizontal.from_unpacked(row) for row in transpose(mat)
        ]
        return TowerAlgebra(
            self.F, self.F_horizontal, self.F_vertical, horizontal_elems
        )

    def __add__(self, other: "TowerAlgebra") -> "TowerAlgebra":
        assert isinstance(other, TowerAlgebra)
        assert (
            self.F == other.F
            and self.F_vertical == other.F_vertical
            and self.F_horizontal == other.F_horizontal
        )
        elems = [e1 + e2 for e1, e2 in zip(self.elems, other.elems)]
        return TowerAlgebra(self.F, self.F_vertical, self.F_horizontal, elems)

    __sub__ = __add__

    def __neg__(self):
        return self

    __mul__ = scale_vertical
