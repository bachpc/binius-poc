from .binary_fields import (
    BinaryField,
    BinaryFieldElement,
    BF1,
    BF2,
    BF4,
    BF8,
    BF16,
    BF32,
    BF64,
    BF128,
)
from .tower_algebra import TowerAlgebra
from .additive_ntt import AdditiveNTT
from .reed_solomon import ReedSolomonCode
from .challenger import Challenger
from .merkle import MerkleTreeVCS
from .multilinear import MultilinearExtension, MultilinearQuery
from .utils import (
    log2,
    tensor_product,
    inner_product,
    vector_multiply_matrix,
    matrix_multiply_vector,
    transpose,
)
from .base_pcs import BaseCommitment, BaseCommitted, BaseProof, BasePCS

from typing import TypeVar

T = TypeVar("T")
Vector = list[T]
Matrix = Vector[Vector[T]]
