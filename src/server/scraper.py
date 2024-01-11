from typing import Optional

from bs4 import BeautifulSoup


def extract_article(html: str) -> Optional[str]:
    """
    Find the longest paragraph in given HTML of a website.
    :param html:
    :return: article text or None if no article was found
    """
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all(name="p", string=True)
    if not tags:
        return None
    texts = [tag.text for tag in tags]
    article = max(texts, key=len)
    return article
