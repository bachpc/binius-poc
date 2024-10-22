from common import (
    BinaryField,
    BinaryFieldElement,
    TowerAlgebra,
    MultilinearExtension,
    MultilinearQuery,
    ReedSolomonCode,
    MerkleTreeVCS,
    Challenger,
    BaseCommitment,
    BaseCommitted,
    BaseProof,
    BasePCS,
    Vector,
    Matrix,
    transpose,
    log2,
)

from dataclasses import dataclass


class BiniusBlockPCS(BasePCS):

    @dataclass
    class Commitment(BaseCommitment):
        vcs_commitment: MerkleTreeVCS.Commitment

        def serialize(self) -> bytes:
            return self.vcs_commitment.serialize()

    @dataclass
    class Committed(BaseCommitted):
        vcs_committed: MerkleTreeVCS.Committed
        encoded_cols: Matrix[BinaryFieldElement]

    @dataclass
    class Proof(BaseProof):
        t_prime: MultilinearExtension
        vcs_proofs: list[tuple[Vector[BinaryFieldElement], MerkleTreeVCS.Proof]]

    def __init__(
        self,
        F: BinaryField,
        FA: BinaryField,
        FE: BinaryField,
        n_vars: int,
        log_rows: int,
        log_inv_rate: int,
        n_challenges: int,
    ):
        assert (
            isinstance(F, BinaryField)
            and isinstance(FA, BinaryField)
            and isinstance(FE, BinaryField)
            and FA.is_extension_of(F)
            and FE.is_extension_of(F)
        )

        self.F = F
        self.FA = FA
        self.FE = FE
        self.n_vars = n_vars
        self.log_rows = log_rows
        self.n_challenges = n_challenges

        self.FE_degree = self.FE.degree(self.F)
        self.FA_degree = self.FA.degree(self.F)
        self.log_cols = self.n_vars - self.log_rows

        assert self.log_cols - log2(self.FA_degree) + log_inv_rate <= self.FA.bit_length
        self.code = ReedSolomonCode(
            self.log_cols - log2(self.FA_degree), log_inv_rate, self.FA
        )
        self.vcs = MerkleTreeVCS(self.code.log_length)

    def commit(self, poly: MultilinearExtension) -> tuple[Commitment, Committed]:
        assert poly.field == self.F and poly.n_vars == self.n_vars

        row_length = 1 << self.log_cols
        evals = poly.evals
        mat = [evals[i : i + row_length] for i in range(0, len(evals), row_length)]

        mat = [self.FA.cast_slice(row) for row in mat]
        encoded_mat = [self.code.encode(row) for row in mat]

        encoded_cols = transpose(encoded_mat)
        vcs_commitment, vcs_committed = self.vcs.commit(encoded_cols)

        commitment = self.Commitment(vcs_commitment)
        committed = self.Committed(vcs_committed, encoded_cols)

        return commitment, committed

    def prove_evaluation(
        self,
        challenger: Challenger,
        committed: Committed,
        poly: MultilinearExtension,
        query: list[BinaryFieldElement],
    ) -> Proof:
        assert poly.field == self.F and poly.n_vars == self.n_vars == len(query)
        assert all(self.FE.check_element(e) for e in query)

        high_partial_query = MultilinearQuery.with_full_query(
            query[self.log_cols :], self.FE
        )
        t_prime = poly.evaluate_partial_high(high_partial_query)

        challenger.observe_slice(t_prime.evals)
        challenges = [
            challenger.sample_bits(self.vcs.log_len) for _ in range(self.n_challenges)
        ]

        merkle_proofs = [
            (
                committed.encoded_cols[index],
                self.vcs.prove_opening(committed.vcs_committed, index),
            )
            for index in challenges
        ]
        proof = self.Proof(t_prime, merkle_proofs)
        return proof

    def check_proof(self, proof: Proof) -> bool:
        return proof.t_prime.field == self.FE and proof.t_prime.n_vars == self.log_cols

    def verify_evaluation(
        self,
        challenger: Challenger,
        commitment: Commitment,
        query: list[BinaryFieldElement],
        proof: Proof,
        value: BinaryFieldElement,
    ) -> bool:
        assert len(query) == self.n_vars
        assert all(self.FE.check_element(e) for e in query)
        assert self.check_proof(proof)

        low_partial_query = MultilinearQuery.with_full_query(
            query[: self.log_cols], self.FE
        )
        computed_value = proof.t_prime.evaluate(low_partial_query)
        if computed_value != value:
            return False

        high_partial_query = MultilinearQuery.with_full_query(
            query[self.log_cols :], self.FE
        )

        t_prime = proof.t_prime.evals
        t_prime = [
            TowerAlgebra(
                self.F, self.FE, self.FA, t_prime[i : i + self.FA_degree]
            ).transpose()
            for i in range(0, len(t_prime), self.FA_degree)
        ]
        u_prime = self.code.encode(t_prime)

        challenger.observe_slice(proof.t_prime.evals)
        challenges = [
            challenger.sample_bits(self.vcs.log_len) for _ in range(self.n_challenges)
        ]

        for index, (col, vcs_proof) in zip(challenges, proof.vcs_proofs):
            if not self.vcs.verify_opening(
                commitment.vcs_commitment, index, vcs_proof, col
            ):
                return False

            lhs = sum(
                (
                    TowerAlgebra.from_tensor(self.F, self.FA, self.FE, x, y)
                    for x, y in zip(col, high_partial_query.expansion())
                ),
                TowerAlgebra.zero(self.F, self.FA, self.FE),
            )

            if lhs != u_prime[index]:
                return False

        return True


if __name__ == "__main__":
    from common.binary_fields import *
    import random
    from copy import deepcopy

    random.seed(123)

    F, FA, FE = BF8, BF32, BF128
    n_vars, log_rows, log_inv_rate, n_challenges = 11, 3, 2, 64

    pcs = BiniusBlockPCS(F, FA, FE, n_vars, log_rows, log_inv_rate, n_challenges)
    print(pcs.code)
    poly = MultilinearExtension.from_evals(
        [F.random_element() for _ in range(1 << n_vars)], F
    )
    query = [FE.random_element() for _ in range(n_vars)]
    value = poly.evaluate(MultilinearQuery.with_full_query(query, FE))

    commitment, committed = pcs.commit(poly)

    challenger = Challenger()
    challenger.observe(commitment.serialize())

    prover_challenger, verifier_challenger = deepcopy(challenger), deepcopy(challenger)
    proof = pcs.prove_evaluation(prover_challenger, committed, poly, query)
    assert pcs.verify_evaluation(verifier_challenger, commitment, query, proof, value)
    print("ok")
