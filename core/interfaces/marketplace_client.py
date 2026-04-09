from abc import ABC, abstractmethod
from typing import Any, Dict


class MarketplaceClient(ABC):
    @abstractmethod
    def get_item_by_legacy_id(self, legacy_item_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_items_by_item_group(self, item_group_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def search_by_epid(self, epid: str, limit: int = 5) -> Dict[str, Any]:
        pass
