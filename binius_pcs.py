from binary_fields import BinaryField, BinaryFieldElement, BF1, BF2, BF4, BF8, BF16, BF32, BF64, BF128
from binary_ntt import reed_solomon_encode
from merkle import hash, merklize, get_root, get_branch, verify_branch
from utils import multilinear_poly_eval, evaluation_tensor_product, log2, transpose
import itertools


def choose_row_length_and_count(log_evaluation_count):
    log_row_length = (log_evaluation_count + 2) // 2
    log_row_count = (log_evaluation_count - 1) // 2
    row_length = 1 << log_row_length
    row_count = 1 << log_row_count
    return log_row_length, log_row_count, row_length, row_count


# pack or unpack a matrix of F1's elements into matrix of F2's elements
def pack_or_unpack_mat(mat, F1, F2):
    assert isinstance(F1, BinaryField) and isinstance(F2, BinaryField)
    assert all(isinstance(v, BinaryFieldElement) and v.field == F1 for v in itertools.chain(*mat))
    if F2.is_extension_of(F1):
        pack_width = F2.pack_width(F1)
        result = [
            [BinaryFieldElement.pack_from(row[i:i + pack_width], F2)
            for i in range(0, len(row), pack_width)]
            for row in mat
        ]
    else:
        result = [
            list(itertools.chain(*(ele.unpack_into(F2) for ele in row)))
            for row in mat
        ]
    return result


def vector_dot_product(v1, v2, field):
    assert len(v1) == len(v2)
    assert isinstance(field, BinaryField)
    assert all(isinstance(v, BinaryFieldElement) and field.is_extension_of(v.field) for v in v1)
    assert all(isinstance(v, BinaryFieldElement) and field.is_extension_of(v.field) for v in v2)
    return sum(
            (
                x * y for x, y in zip(v1, v2)
            ),
            field.ZERO
        )


def vector_matrix_product(vec, mat, field, left_multiply=True):
    # assert isinstance(field, BinaryField)
    # assert all(isinstance(v, BinaryFieldElement) and v.field == field for v in itertools.chain(*mat))
    # assert all(isinstance(v, BinaryFieldElement) and v.field == field for v in vec)
    # assert len(mat[0]) == len(vec)
    # assert all(len(mat[i]) == len(mat[0]) for i in range(1, len(mat)))
    if left_multiply:
        mat = transpose(mat)
    return [vector_dot_product(row, vec, field) for row in mat]


# `F`: The base field type of committed elements.
# `FA`: The field type of the encoding alphabet.
# `FI`: The intermediate field type that base field elements are packed into.
# `FE`: The extension field type used for cryptographic challenges.
class BiniusPCS:

    def __init__(self, F, FA, FI, FE, inv_rate=2, num_challenges=32):
        assert isinstance(F, BinaryField) \
            and isinstance(FA, BinaryField) \
            and isinstance(FI, BinaryField) \
            and isinstance(FE, BinaryField)

        assert FI.is_extension_of(F) \
            and FI.is_extension_of(FA) \
            and FE.is_extension_of(F) \
            and FE.is_extension_of(FI)

        self.F = F
        self.FA = FA
        self.FI = FI
        self.FE = FE
        self.inv_rate = inv_rate
        self.num_challenges = num_challenges

        self.width_FI = self.FI.pack_width(self.F)
        self.width_FI_FA = self.FI.pack_width(self.FA)
        self.width_FE = self.FE.pack_width(self.F)
        self.width_FE_FI = self.FE.pack_width(self.FI)

    # Construction 3.7. In this case, the encoding alphabet is a subfield of
    # the polynomial's coefficient field.
    @classmethod
    def newBasicPCS(cls, F, FA, FE, inv_rate=2, num_challenges=32):
        return cls(F, FA, F, FE, inv_rate, num_challenges)

    # Construction 3.11. In this case, the encoding alphabet is an extension field of
    # the polynomial's coefficient field.
    @classmethod
    def newBlockPCS(cls, F, FA, FE, inv_rate=2, num_challenges=32):
        return cls(F, FA, FA, FE, inv_rate, num_challenges)

    # `poly` is represented by its evaluations over hypercube
    def commit(self, poly):
        assert all(isinstance(v, BinaryFieldElement) for v in poly)
        assert all(v.field == self.F for v in poly)

        log_row_length, log_row_count, row_length, row_count = \
            choose_row_length_and_count(log2(len(poly)))
        extended_row_length = row_length // self.width_FI * self.inv_rate
        # print(f'{extended_row_length = }')

        rows = [
            poly[i:i + row_length]
            for i in range(0, len(poly), row_length)
        ]

        rows = pack_or_unpack_mat(rows, self.F, self.FA)

        encoded_mat = [
            reed_solomon_encode(self.FA, row, self.inv_rate)
            for row in rows
        ]
        encoded_mat = pack_or_unpack_mat(encoded_mat, self.FA, self.FI)

        assert all(len(row) == extended_row_length for row in encoded_mat)

        columns = transpose(encoded_mat)
        packed_columns = [
            b'@'.join(ele.to_bytes((self.FI.bit_length + 7) // 8, 'little') for ele in col)
            for col in columns
        ]

        merkle_tree = merklize(packed_columns)
        root = get_root(merkle_tree)

        commitment = {
            'root': root,
        }
        committed = {
            'merkle_tree': merkle_tree,
            'encoded_mat': encoded_mat,
        }

        return commitment, committed

    # `poly` is represented by its evaluations over hypercube
    def prove_evaluation(self, committed, poly, query):
        assert 1 << len(query) == len(poly)
        assert all(isinstance(v, BinaryFieldElement) for v in poly)
        assert all(isinstance(v, BinaryFieldElement) for v in query)
        assert all(v.field == self.F for v in poly)
        assert all(v.field == self.FE for v in query)

        merkle_tree, encoded_mat = committed['merkle_tree'], committed['encoded_mat']

        log_row_length, log_row_count, row_length, row_count = \
            choose_row_length_and_count(log2(len(poly)))
        extended_row_length = row_length // self.width_FI * self.inv_rate

        assert all(len(row) == extended_row_length for row in encoded_mat)

        rows = [
            poly[i:i + row_length]
            for i in range(0, len(poly), row_length)
        ]

        high_partial_query_tensor = evaluation_tensor_product(query[log_row_length:], self.FE)
        assert len(high_partial_query_tensor) == len(rows) == row_count

        t_prime = vector_matrix_product(high_partial_query_tensor, rows, self.FE, True)

        h = hash(b'@'.join(ele.to_bytes((self.FE.bit_length + 7) // 8, 'little') for ele in t_prime))
        challenges = [
            int.from_bytes(hash(h + bytes([i])), 'little') % extended_row_length
            for i in range(self.num_challenges)
        ]

        columns = transpose(encoded_mat)
        merkle_proofs = [
            (columns[c], get_branch(merkle_tree, c))
            for c in challenges
        ]

        proof = {
            'eval': multilinear_poly_eval(poly, query, self.FE),
            't_prime': t_prime,
            'merkle_proofs': merkle_proofs,
        }

        return proof

    def verify_evaluation(self, commitment, query, proof):
        assert all(isinstance(v, BinaryFieldElement) for v in query)
        assert all(v.field == self.FE for v in query)

        root = commitment['root']
        t_prime, merkle_proofs, value = proof['t_prime'], proof['merkle_proofs'], proof['eval']

        assert isinstance(value, BinaryFieldElement) and value.field == self.FE
        assert all(isinstance(v, BinaryFieldElement) for v in t_prime)
        assert all(v.field == self.FE for v in t_prime)

        log_row_length, log_row_count, row_length, row_count = \
            choose_row_length_and_count(len(query))
        extended_row_length = row_length // self.width_FI * self.inv_rate

        h = hash(b'@'.join(ele.to_bytes((self.FE.bit_length + 7) // 8, 'little') for ele in t_prime))
        challenges = [
            int.from_bytes(hash(h + bytes([i])), 'little') % extended_row_length
            for i in range(self.num_challenges)
        ]

        for challenge, (column, merkle_proof) in zip(challenges, merkle_proofs):
            packed_column = b'@'.join(ele.to_bytes((self.FI.bit_length + 7) // 8, 'little') for ele in column)
            if not verify_branch(root, challenge, packed_column, merkle_proof):
                return False

        low_partial_query_tensor = evaluation_tensor_product(query[:log_row_length], self.FE)
        assert len(low_partial_query_tensor) == len(t_prime) == row_length

        computed_value = vector_dot_product(low_partial_query_tensor, t_prime, self.FE)
        if computed_value != value:
            return False

        u_prime = self.encode_ext(t_prime)
        assert len(u_prime) == extended_row_length

        high_partial_query_tensor = evaluation_tensor_product(query[log_row_length:], self.FE)

        for challenge, (column, _) in zip(challenges, merkle_proofs):
            u_mat = [ele.unpack_into(self.F) for ele in column]
            computed_value = vector_matrix_product(high_partial_query_tensor, u_mat, self.FE, True)
            # print(computed_value)
            # print(u_prime[challenge])
            # print()
            if computed_value != u_prime[challenge]:
                return False

        return True

    def encode_ext(self, t_prime):
        assert all(isinstance(v, BinaryFieldElement) for v in t_prime)
        assert all(v.field == self.FE for v in t_prime)

        mat = [ele.unpack_into(self.F) for ele in t_prime]
        mat = transpose(mat)

        mat = pack_or_unpack_mat(mat, self.F, self.FA)
        encoded_mat = [
            reed_solomon_encode(self.FA, row, self.inv_rate)
            for row in mat
        ]
        encoded_mat = pack_or_unpack_mat(encoded_mat, self.FA, self.F)

        encoded_mat = transpose(encoded_mat)
        u_prime_flat = [BinaryFieldElement.pack_from(row, self.FE) for row in encoded_mat]

        # Pack into tower algebra elements
        u_prime = [u_prime_flat[i:i + self.width_FI] for i in range(0, len(u_prime_flat), self.width_FI)]

        return u_prime


if __name__ == '__main__':
    import random
    random.seed(123)

    F, FA, FE = BF8, BF8, BF128
    nvars = 10

    pcs = BiniusPCS.newBasicPCS(F, FA, FE, inv_rate=16)
    poly = [F.random_element() for _ in range(1 << nvars)]
    query = [FE.random_element() for _ in range(nvars)]

    commitment, committed = pcs.commit(poly)
    proof = pcs.prove_evaluation(committed, poly, query)
    assert pcs.verify_evaluation(commitment, query, proof)

    ###
    F, FA, FE = BF1, BF16, BF128
    nvars = 12

    pcs = BiniusPCS.newBlockPCS(F, FA, FE, inv_rate=16)
    poly = [F.random_element() for _ in range(1 << nvars)]
    query = [FE.random_element() for _ in range(nvars)]

    commitment, committed = pcs.commit(poly)
    proof = pcs.prove_evaluation(committed, poly, query)
    assert pcs.verify_evaluation(commitment, query, proof)
