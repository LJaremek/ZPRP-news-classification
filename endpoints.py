import random
from typing import Annotated

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import scraper

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def mock_rate_article(article_text: str) -> float:
    return random.random()


async def extract_article(url):
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
    return article


class ArticleRating(BaseModel):
    article: str
    rating: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/rate_article", response_class=HTMLResponse)
async def rate_article_form(request: Request, article_text: Annotated[str, Form()]):
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "text": article_text,
            "rating_percent": int(mock_rate_article(article_text) * 100),
        },
    )


@app.post("/rate_url", response_class=HTMLResponse)
async def rate_url_form(request: Request, article_url: Annotated[str, Form()]):
    article_text = await extract_article(article_url)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "text": article_text,
            "rating_percent": int(mock_rate_article(article_text) * 100),
        },
    )


@app.get("/rate_article")
async def rate_article(article: str):
    return ArticleRating(article=article, rating=mock_rate_article(article))


@app.get("/rate_url")
async def rate_url(url: str):
    article = await extract_article(url)
    return ArticleRating(article=article, rating=mock_rate_article(article))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, ssl_keyfile=None, ssl_certfile=None)
