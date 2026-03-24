from pprint import pprint
from dataclasses import asdict

from infrastructure.external_clients.ebay.ebay_auth_client import EbayAuthClient
from infrastructure.external_clients.ebay.ebay_browse_client import EbayBrowseClient
from infrastructure.external_clients.ebay.ebay_url_parser import EbayUrlParser
from core.services.normalization_service import NormalizationService


def main():
    url_a = "https://www.ebay.com.sg/itm/206158794969?itmmeta=01KMF0940PYQKEP45AJ0BP7K4M&hash=item30000590d9:g:JnYAAeSwiEVpwANm"
    url_b = "https://www.ebay.com.sg/p/4062765295?iid=377055098797"

    auth_client = EbayAuthClient()
    access_token = auth_client.get_access_token()

    browse_client = EbayBrowseClient(access_token=access_token, marketplace_id="EBAY_SG")
    url_parser = EbayUrlParser()

    normalization_service = NormalizationService(
        marketplace_client=browse_client,
        url_parser=url_parser,
    )

    candidate_a = normalization_service.normalize(url_a)
    candidate_b = normalization_service.normalize(url_b)

    pprint(asdict(candidate_a))
    print("\n" + "=" * 80 + "\n")
    pprint(asdict(candidate_b))


if __name__ == "__main__":
    main()