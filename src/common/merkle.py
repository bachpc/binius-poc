from .binary_fields import BinaryFieldElement
from .base_pcs import BaseCommitment, BaseCommitted, BaseProof

from hashlib import sha256
from dataclasses import dataclass


class MerkleTreeVCS:

    @dataclass
    class Commitment(BaseCommitment):
        merkle_root: bytes

        def serialize(self) -> bytes:
            return self.merkle_root

    @dataclass
    class Committed(BaseCommitted):
        merkle_tree: list[bytes]

    @dataclass
    class Proof(BaseProof):
        branch: list[bytes]

    def __init__(self, log_len: int):
        self.log_len = log_len

    def _hash(self, vec: list[BinaryFieldElement]) -> bytes:
        h = sha256()
        for v in vec:
            h.update(v.to_bytes())
        return h.digest()

    def _compress(self, x: bytes, y: bytes) -> bytes:
        return sha256(x + y).digest()

    def commit(
        self, vecs: list[list[BinaryFieldElement]]
    ) -> tuple[Commitment, Committed]:
        assert len(vecs) == 1 << self.log_len
        tree = [None] * len(vecs) + [self._hash(vec) for vec in vecs]
        for i in range(len(vecs) - 1, 0, -1):
            tree[i] = self._compress(tree[i << 1], tree[(i << 1) ^ 1])
        commitment = MerkleTreeVCS.Commitment(tree[1])
        committed = MerkleTreeVCS.Committed(tree)
        return commitment, committed

    def prove_opening(self, committed: Committed, index: int) -> Proof:
        tree = committed.merkle_tree
        offset_pos = int(index + len(tree) // 2)
        branch = [tree[(offset_pos >> i) ^ 1] for i in range(self.log_len)]
        return MerkleTreeVCS.Proof(branch)

    def verify_opening(
        self,
        commitment: Commitment,
        index: int,
        proof: Proof,
        values: list[BinaryFieldElement],
    ) -> bool:
        root = self._hash(values)
        for b in proof.branch:
            if index & 1:
                root = self._compress(b, root)
            else:
                root = self._compress(root, b)
            index >>= 1
        return root == commitment.merkle_root


if __name__ == "__main__":

    from binary_fields import *

    log_len = 8
    vcs = MerkleTreeVCS(log_len)

    data = [[BF128.random_element() for _ in range(200)] for _ in range(1 << log_len)]
    index = 123

    commitment, committed = vcs.commit(data)
    proof = vcs.prove_opening(committed, index)
    assert vcs.verify_opening(commitment, index, proof, data[index])
