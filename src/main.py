from common import *
from binius import BiniusBasicPCS, BiniusBlockPCS
from fri_binius import RingSwitchingPCS

import random
from copy import deepcopy


def testBiniusBasicPCS(seed=123):
    random.seed(seed)
    K, L = BF8, BF128
    n_vars, log_rows, log_inv_rate, n_challenges = 11, 5, 2, 64

    pcs = BiniusBasicPCS(K, L, n_vars, log_rows, log_inv_rate, n_challenges)
    print(pcs.code)
    poly = MultilinearExtension.from_evals(
        [K.random_element() for _ in range(1 << n_vars)], K
    )
    query = [L.random_element() for _ in range(n_vars)]
    value = poly.evaluate(MultilinearQuery.with_full_query(query, L))
    challenger = Challenger()

    commitment, committed = pcs.commit(poly)
    challenger.observe(commitment.serialize())

    prover_challenger, verifier_challenger = deepcopy(challenger), deepcopy(challenger)
    proof = pcs.prove_evaluation(prover_challenger, committed, poly, query)
    assert pcs.verify_evaluation(verifier_challenger, commitment, query, proof, value)
    print("testBiniusBasicPCS ok")


def testBiniusBlockPCS(seed=123):
    random.seed(seed)
    F, FA, FE = BF8, BF32, BF128
    n_vars, log_rows, log_inv_rate, n_challenges = 11, 5, 2, 64

    pcs = BiniusBlockPCS(F, FA, FE, n_vars, log_rows, log_inv_rate, n_challenges)
    print(pcs.code)
    poly = MultilinearExtension.from_evals(
        [F.random_element() for _ in range(1 << n_vars)], F
    )
    query = [FE.random_element() for _ in range(n_vars)]
    value = poly.evaluate(MultilinearQuery.with_full_query(query, FE))
    challenger = Challenger()

    commitment, committed = pcs.commit(poly)
    challenger.observe(commitment.serialize())

    prover_challenger, verifier_challenger = deepcopy(challenger), deepcopy(challenger)
    proof = pcs.prove_evaluation(prover_challenger, committed, poly, query)
    assert pcs.verify_evaluation(verifier_challenger, commitment, query, proof, value)
    print("testBiniusBlockPCS ok")


def testRingSwitchingPCS(seed=123):
    random.seed(seed)
    K, L = BF8, BF128
    n_vars = 11
    log_rows, log_inv_rate, n_challenges = 3, 2, 64

    inner_pcs = BiniusBasicPCS(
        L, L, n_vars - log2(L.degree(K)), log_rows, log_inv_rate, n_challenges
    )
    pcs = RingSwitchingPCS(K, L, inner_pcs, n_vars)
    poly = MultilinearExtension.from_evals(
        [K.random_element() for _ in range(1 << n_vars)], K
    )
    query = [L.random_element() for _ in range(n_vars)]
    value = poly.evaluate(MultilinearQuery.with_full_query(query, L))
    challenger = Challenger()

    commitment, committed = pcs.commit(poly)
    challenger.observe(commitment.serialize())

    prover_challenger, verifier_challenger = deepcopy(challenger), deepcopy(challenger)
    proof = pcs.prove_evaluation(prover_challenger, committed, poly, query)
    assert pcs.verify_evaluation(verifier_challenger, commitment, query, proof, value)
    print("testRingSwitchingPCS ok")


if __name__ == "__main__":
    testBiniusBasicPCS()
    testBiniusBlockPCS()
    testRingSwitchingPCS()
