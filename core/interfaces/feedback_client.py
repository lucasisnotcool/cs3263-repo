from abc import ABC, abstractmethod
from typing import Any, Dict


class FeedbackClient(ABC):
    @abstractmethod
    def get_feedback(
        self,
        user_id: str,
        feedback_type: str = "FEEDBACK_RECEIVED",
        limit: int = 25,
        offset: int = 0,
        role: str | None = None,
    ) -> Dict[str, Any]:
        pass
