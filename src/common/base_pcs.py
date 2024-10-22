from .binary_fields import BinaryFieldElement
from .multilinear import MultilinearExtension
from .challenger import Challenger

from abc import ABC, abstractmethod


class BaseCommitment(ABC):

    @abstractmethod
    def serialize(self) -> bytes:
        pass


class BaseCommitted(ABC):
    pass


class BaseProof(ABC):
    pass


class BasePCS(ABC):

    @abstractmethod
    def commit(
        self, poly: MultilinearExtension
    ) -> tuple[BaseCommitment, BaseCommitted]:
        pass

    @abstractmethod
    def prove_evaluation(
        self,
        challenger: Challenger,
        committed: BaseCommitted,
        poly: MultilinearExtension,
        query: list[BinaryFieldElement],
    ) -> BaseProof:
        pass

    @abstractmethod
    def verify_evaluation(
        self,
        challenger: Challenger,
        commitment: BaseCommitment,
        query: list[BinaryFieldElement],
        proof: BaseProof,
        value: BinaryFieldElement,
    ) -> bool:
        pass
