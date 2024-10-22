from common import (
    BinaryField,
    BinaryFieldElement,
    TowerAlgebra,
    MultilinearExtension,
    MultilinearQuery,
)

from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SumcheckClaim:
    eval_point: list[BinaryFieldElement]
    eval: TowerAlgebra


@dataclass
class RoundClaim:
    partial_point: list[BinaryFieldElement]
    current_round_sum: TowerAlgebra


@dataclass
class RoundProof:
    coeffs: list[TowerAlgebra]


@dataclass
class ReducedClaim:
    eval_point: list[BinaryFieldElement]
    eval: TowerAlgebra


def reduce_round_claim(
    z_i: BinaryFieldElement,
    claim: RoundClaim,
    challenge: BinaryFieldElement,
    proof: RoundProof,
) -> RoundClaim:
    linear_term = proof.coeffs[0]
    constant_term = claim.current_round_sum - linear_term.scale_vertical(z_i)
    new_round_sum = constant_term + linear_term.scale_horizontal(challenge)
    return RoundClaim(claim.partial_point + [challenge], new_round_sum)


class SumcheckProver:

    def __init__(
        self,
        K: BinaryField,
        L: BinaryField,
        claim: SumcheckClaim,
        witness: MultilinearExtension,
    ):
        assert (
            isinstance(K, BinaryField)
            and isinstance(L, BinaryField)
            and L.is_extension_of(K)
        )
        assert all(
            isinstance(e, BinaryFieldElement) for e in claim.eval_point
        ) and isinstance(claim.eval, TowerAlgebra)
        assert len(claim.eval_point) == witness.n_vars
        assert witness.field == L

        self.K = K
        self.L = L
        self.n_vars = witness.n_vars
        self.round = 0
        self.eval_point = claim.eval_point[:]
        self.round_claim = RoundClaim([], claim.eval.copy())
        self.last_round_proof: RoundProof = None
        self.eq_ind = MultilinearQuery.with_full_query(
            claim.eval_point[1:], self.L
        ).expansion()
        self.multilinear_ind = witness.copy()

    def fold_eq_ind(self):
        self.eq_ind = [
            self.eq_ind[i] + self.eq_ind[i + 1] for i in range(0, len(self.eq_ind), 2)
        ]

    def reduce_claim(self, prev_rd_challenge: BinaryFieldElement):
        assert self.round > 0 and isinstance(prev_rd_challenge, BinaryFieldElement)
        new_round_claim = reduce_round_claim(
            self.eval_point[self.round - 1],
            self.round_claim,
            prev_rd_challenge,
            self.last_round_proof,
        )
        self.round_claim = new_round_claim

    def fold_multilinear_ind(self, prev_rd_challenge: BinaryFieldElement):
        assert isinstance(prev_rd_challenge, BinaryFieldElement)
        partial_query = MultilinearQuery.with_full_query([prev_rd_challenge], self.L)
        self.multilinear_ind = self.multilinear_ind.evaluate_partial_low(partial_query)

    def execute_round(self, prev_rd_challenge: BinaryFieldElement) -> RoundProof:
        assert (self.round == 0 and prev_rd_challenge is None) or (
            self.round > 0 or prev_rd_challenge
        )
        assert self.round < self.n_vars

        if prev_rd_challenge:
            self.fold_multilinear_ind(prev_rd_challenge)
            self.fold_eq_ind()
            self.reduce_claim(prev_rd_challenge)

        rd_vars = self.n_vars - self.round
        eq_ind = MultilinearExtension.from_evals(self.eq_ind, self.L)
        # print(eq_ind.n_vars, self.multilinear_ind.n_vars, rd_vars)
        eval_1 = sum(
            (
                TowerAlgebra.from_tensor(
                    self.K,
                    self.L,
                    self.L,
                    eq_ind.evaluate_on_hypercube(i),
                    self.multilinear_ind.evaluate_on_hypercube(i << 1 | 1),
                )
                for i in range(1 << (rd_vars - 1))
            ),
            TowerAlgebra.zero(self.K, self.L, self.L),
        )

        z_i = self.eval_point[self.round]
        denominator = (self.L.ONE - z_i).inv()
        eval_0 = (
            self.round_claim.current_round_sum - eval_1.scale_vertical(z_i)
        ).scale_vertical(denominator)
        coeffs = [eval_1 - eval_0]

        round_proof = RoundProof(coeffs)
        self.last_round_proof = deepcopy(round_proof)
        self.round += 1
        return round_proof

    def finalize(self, prev_rd_challenge: BinaryFieldElement) -> ReducedClaim:
        assert self.round == self.n_vars and prev_rd_challenge
        self.reduce_claim(prev_rd_challenge)
        return ReducedClaim(
            self.round_claim.partial_point,
            self.round_claim.current_round_sum,
        )
