import re
import urllib.parse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedEbayUrl:
    page_type: str
    legacy_item_id: Optional[str] = None
    epid: Optional[str] = None


class EbayUrlParser:
    def parse(self, url: str) -> ParsedEbayUrl:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        match_itm = re.search(r"/itm/(\d+)", path)
        if match_itm:
            return ParsedEbayUrl(
                page_type="listing",
                legacy_item_id=match_itm.group(1),
            )

        match_p = re.search(r"/p/(\d+)", path)
        if match_p:
            return ParsedEbayUrl(
                page_type="product_page",
                legacy_item_id=query.get("iid", [None])[0],
                epid=match_p.group(1),
            )

        return ParsedEbayUrl(page_type="unknown")