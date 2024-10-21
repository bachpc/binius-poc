from binary_fields import BinaryField, BinaryFieldElement
from tower_algebra import TowerAlgebra


class AdditiveNTT:

    def __init__(self, log_degree: int, log_domain_size: int, field: BinaryField):
        assert isinstance(field, BinaryField)
        self.log_degree = log_degree
        self.log_domain_size = log_domain_size
        self.field = field
        self._precompute()

    def _precompute(self):
        norms = [self.field.ONE]
        s_evals = [[self.field(1 << i) for i in range(self.log_domain_size)]]
        for _ in range(1, self.log_domain_size):
            norm_prev = norms[-1]
            s_evals_prev = s_evals[-1]
            norms.append(s_evals_prev[0] * (s_evals_prev[0] + norm_prev))
            s_evals.append([e * (e + norm_prev) for e in s_evals_prev])
        for i in range(self.log_domain_size):
            for j in range(len(s_evals[i])):
                s_evals[i][j] = s_evals[i][j] / norms[i]

        s_evals_expanded = []
        for i in range(self.log_domain_size):
            expanded = [self.field.ZERO]
            for e in s_evals[i]:
                expanded.extend([expanded_prev + e for expanded_prev in expanded])
            s_evals_expanded.append(expanded)

        self.s_evals = s_evals_expanded

    def _get_twiddle(self, i: int, u: int) -> BinaryFieldElement:
        return self.s_evals[i][u]

    def forward_transform(
        self, data: list[BinaryFieldElement] | list[TowerAlgebra]
    ) -> list[BinaryFieldElement] | list[TowerAlgebra]:
        assert len(data) == 1 << self.log_domain_size
        assert all(isinstance(v, BinaryFieldElement) for v in data) or all(
            isinstance(v, TowerAlgebra) for v in data
        )

        for i in range(self.log_degree - 1, -1, -1):
            for u in range(1 << (self.log_domain_size - i - 1)):
                twiddle = self._get_twiddle(i, u)
                for v in range(1 << i):
                    idx0 = u << (i + 1) | v
                    idx1 = idx0 | 1 << i
                    data[idx0] += data[idx1].__mul__(twiddle)
                    data[idx1] += data[idx0]

        return data

    def inverse_transform(
        self, data: list[BinaryFieldElement] | list[TowerAlgebra]
    ) -> list[BinaryFieldElement] | list[TowerAlgebra]:
        assert len(data) == 1 << self.log_domain_size
        assert all(isinstance(v, BinaryFieldElement) for v in data) or all(
            isinstance(v, TowerAlgebra) for v in data
        )

        for i in range(self.log_degree):
            for u in range(1 << (self.log_domain_size - i - 1)):
                twiddle = self._get_twiddle(i, u)
                for v in range(1 << i):
                    idx0 = u << (i + 1) | v
                    idx1 = idx0 | 1 << i
                    data[idx1] += data[idx0]
                    data[idx0] += data[idx1].__mul__(twiddle)

        return data


if __name__ == "__main__":

    from binary_fields import *
    from copy import deepcopy

    field = BF128
    log_degree = 5
    log_domain_size = 6

    data = [field.random_element() for _ in range(1 << log_degree)]
    data = sum([deepcopy(data) for _ in range(1 << (log_domain_size - log_degree))], [])
    print(data)

    ntt = AdditiveNTT(log_degree, log_domain_size, field)
    data2 = ntt.forward_transform(data)
    print(data2)
    print(data == ntt.inverse_transform(data2))
