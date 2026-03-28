from abc import ABC, abstractmethod
from core.entities.candidate import Candidate


class Normalizer(ABC):
    @abstractmethod
    def normalize(self, url: str) -> Candidate:
        pass