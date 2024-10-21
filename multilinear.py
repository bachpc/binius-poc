from binary_fields import BinaryField, BinaryFieldElement
from utils import log2, inner_product, vector_multiply_matrix, matrix_multiply_vector


class MultilinearQuery:

    def __init__(self, field: BinaryField):
        assert isinstance(field, BinaryField)
        self.field = field
        self.n_vars = 0
        self.expanded_query = [field(1)]

    def __repr__(self) -> str:
        return f"MultilinearQuery(n_vars={self.n_vars}) in {self.field}"

    @classmethod
    def with_full_query(
        cls, query: list[BinaryFieldElement], field: BinaryField
    ) -> "MultilinearQuery":
        return cls(field).update(query)

    def update(
        self, extra_query_coordinates: list[BinaryFieldElement]
    ) -> "MultilinearQuery":
        assert all(isinstance(e, BinaryFieldElement) for e in extra_query_coordinates)
        new_expanded_query = self.expanded_query
        for coord in extra_query_coordinates:
            p0 = [(self.field.ONE - coord) * v for v in new_expanded_query]
            p1 = [c0 + v for c0, v in zip(p0, new_expanded_query)]
            new_expanded_query = p0 + p1
        self.n_vars += len(extra_query_coordinates)
        self.expanded_query = new_expanded_query
        return self

    def expansion(self) -> list[BinaryFieldElement]:
        return self.expanded_query


class MultilinearExtension:

    def __init__(
        self, n_vars: int, evals: list[BinaryFieldElement], field: BinaryField
    ):
        assert isinstance(field, BinaryField)
        assert all(field.check_element(e) for e in evals)
        assert len(evals) == 1 << n_vars
        self.field = field
        self.n_vars = n_vars
        self.evals = evals

    def __repr__(self) -> str:
        return f"MultilinearExtension(n_vars={self.n_vars}) in {self.field}"

    @classmethod
    def from_evals(
        cls, evals: list[BinaryFieldElement], field: BinaryField
    ) -> "MultilinearExtension":
        n_vars = log2(len(evals))
        return cls(n_vars, evals, field)

    def evaluate(self, query: MultilinearQuery) -> BinaryFieldElement:
        assert query.n_vars == self.n_vars
        return inner_product(query.expansion(), self.evals, query.field | self.field)

    def evaluate_partial_high(self, query: MultilinearQuery) -> "MultilinearExtension":
        assert query.n_vars <= self.n_vars
        row_length = 1 << (self.n_vars - query.n_vars)
        mat = [
            self.evals[i : i + row_length]
            for i in range(0, len(self.evals), row_length)
        ]
        new_evals = vector_multiply_matrix(
            query.expansion(), mat, query.field | self.field
        )
        return MultilinearExtension.from_evals(new_evals, query.field | self.field)

    def evaluate_partial_low(self, query: MultilinearQuery) -> "MultilinearExtension":
        assert query.n_vars <= self.n_vars
        row_length = 1 << query.n_vars
        mat = [
            self.evals[i : i + row_length]
            for i in range(0, len(self.evals), row_length)
        ]
        new_evals = matrix_multiply_vector(
            mat, query.expansion(), query.field | self.field
        )
        return MultilinearExtension.from_evals(new_evals, query.field | self.field)
