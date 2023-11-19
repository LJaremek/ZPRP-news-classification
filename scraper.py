from typing import Optional

from bs4 import BeautifulSoup


def extract_article(page: str) -> Optional[str]:
    soup = BeautifulSoup(page, "html.parser")
    tags = soup.find_all(name="p", string=True)
    if not tags:
        return None
    texts = [tag.text for tag in tags]
    article = max(texts, key=len)
    return article
