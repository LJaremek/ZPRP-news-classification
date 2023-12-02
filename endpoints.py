import random
import uvicorn

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import scraper

app = FastAPI()


def mock_rate_article(article_text: str) -> float:
    return random.random()


class ArticleRating(BaseModel):
    article: str
    rating: float


@app.get("/rate_article")
async def rate_article(article: str):
    return ArticleRating(article=article, rating=mock_rate_article(article))


@app.get("/rate_url")
async def rate_url(url: str):
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="url could not be fetched")

    if response.status_code not in range(200, 299):
        raise HTTPException(
            status_code=400,
            detail=f"invalid status code from fetched url: {response.status_code}",
        )

    page = response.text
    article = scraper.extract_article(page)
    if not article:
        raise HTTPException(status_code=400, detail="could not find article on page")

    return ArticleRating(article=article, rating=mock_rate_article(article))


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0",
        port=8090,
        ssl_keyfile=None,
        ssl_certfile=None
        )