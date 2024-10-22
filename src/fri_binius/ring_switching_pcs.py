from common import (
    BinaryField,
    BinaryFieldElement,
    TowerAlgebra,
    MultilinearExtension,
    MultilinearQuery,
    Challenger,
    BaseCommitment,
    BaseCommitted,
    BaseProof,
    BasePCS,
    log2,
)
from .sumcheck import (
    SumcheckClaim,
    RoundClaim,
    RoundProof,
    ReducedClaim,
    SumcheckProver,
    reduce_round_claim,
)

from dataclasses import dataclass


class RingSwitchingPCS(BasePCS):

    @dataclass
    class Commitment(BaseCommitment):
        inner_pcs_commitment: BaseCommitment

        def serialize(self) -> bytes:
            return self.inner_pcs_commitment.serialize()

    @dataclass
    class Committed(BaseCommitted):
        inner_pcs_committed: BaseCommitted

    @dataclass
    class Proof(BaseProof):
        sumcheck_proof: list[RoundProof]
        sumcheck_eval: TowerAlgebra
        inner_pcs_proof: BaseProof

    def __init__(
        self,
        K: BinaryField,
        L: BinaryField,
        inner_pcs: BasePCS,
        n_vars: int,
    ):
        assert (
            isinstance(K, BinaryField)
            and isinstance(L, BinaryField)
            and L.is_extension_of(K)
        )

        self.K = K
        self.L = L
        self.inner_pcs = inner_pcs
        self.n_vars = n_vars
        self.L_degree = self.L.degree(self.K)

    def commit(self, poly: MultilinearExtension) -> tuple[Commitment, Committed]:
        assert poly.field == self.K and poly.n_vars == self.n_vars

        packed_polys = MultilinearExtension.from_evals(
            self.L.cast_slice(poly.evals), self.L
        )

        inner_commitment, inner_committed = self.inner_pcs.commit(packed_polys)
        commitment = self.Commitment(inner_commitment)
        committed = self.Committed(inner_committed)
        return commitment, committed

    def prove_evaluation(
        self,
        challenger: Challenger,
        committed: Committed,
        poly: MultilinearExtension,
        query: list[BinaryFieldElement],
    ) -> Proof:
        assert poly.field == self.K and poly.n_vars == self.n_vars == len(query)
        assert all(self.L.check_element(e) for e in query)

        packed_polys = MultilinearExtension.from_evals(
            self.L.cast_slice(poly.evals), self.L
        )

        high_query = query[log2(self.L_degree) :]
        expanded_query = MultilinearQuery.with_full_query(high_query, self.L)
        partial_eval = poly.evaluate_partial_high(expanded_query)
        sumcheck_eval = TowerAlgebra.new(self.K, self.L, self.L, partial_eval.evals)

        challenger.observe(sumcheck_eval)

        sumcheck_claim = SumcheckClaim(high_query, sumcheck_eval)
        sumcheck_prover = SumcheckProver(self.K, self.L, sumcheck_claim, packed_polys)
        prev_rd_challenge = None
        rd_proofs = []
        for _ in range(packed_polys.n_vars):
            sumcheck_round = sumcheck_prover.execute_round(prev_rd_challenge)
            challenger.observe_slice(sumcheck_round.coeffs[:])
            prev_rd_challenge = challenger.sample(self.L)
            rd_proofs.append(sumcheck_round)
        reduced_claim = sumcheck_prover.finalize(prev_rd_challenge)

        inner_pcs_proof = self.inner_pcs.prove_evaluation(
            challenger,
            committed.inner_pcs_committed,
            packed_polys,
            reduced_claim.eval_point,
        )

        return self.Proof(rd_proofs, sumcheck_eval, inner_pcs_proof)

    def verify_evaluation(
        self,
        challenger: Challenger,
        commitment: Commitment,
        query: list[BinaryFieldElement],
        proof: Proof,
        value: BinaryFieldElement,
    ) -> bool:
        assert len(query) == self.n_vars
        assert all(self.L.check_element(e) for e in query)

        low_query, high_query = (
            query[: log2(self.L_degree)],
            query[log2(self.L_degree) :],
        )

        sumcheck_eval = proof.sumcheck_eval
        challenger.observe(sumcheck_eval)

        expanded_query = MultilinearQuery.with_full_query(low_query, self.L)
        computed_eval = MultilinearExtension.from_evals(
            sumcheck_eval.elems[:], self.L
        ).evaluate(expanded_query)
        if value != computed_eval:
            return False

        sumcheck_eval, sumcheck_proof = proof.sumcheck_eval, proof.sumcheck_proof
        sumcheck_claim = SumcheckClaim(high_query, sumcheck_eval)

        if len(sumcheck_proof) != len(high_query):
            return False

        rd_claim = RoundClaim([], sumcheck_claim.eval)
        for round, round_proof in enumerate(sumcheck_proof):
            challenger.observe_slice(round_proof.coeffs[:])
            sumcheck_round_challenge = challenger.sample(self.L)
            rd_claim = reduce_round_claim(
                sumcheck_claim.eval_point[round],
                rd_claim,
                sumcheck_round_challenge,
                round_proof,
            )
        reduced_claim = ReducedClaim(rd_claim.partial_point, rd_claim.current_round_sum)
        # print(reduced_claim.eval.transpose())

        try:
            eval = reduced_claim.eval.transpose().try_extract_vertical()
        except:
            return False

        return self.inner_pcs.verify_evaluation(
            challenger,
            commitment.inner_pcs_commitment,
            reduced_claim.eval_point,
            proof.inner_pcs_proof,
            eval,
        )
