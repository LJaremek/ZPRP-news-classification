import random
import time
from typing import Annotated

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from selenium import webdriver
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import scraper


# from src.models.predict import predict
def predict(article_text:str, use_cuda:bool):
    # FIXME remove this when real predict works
    return "true", random.random()

STATIC_PAGE_TIMEOUT = 5
DYNAMIC_PAGE_WAIT = 5
USE_CUDA = False

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def load_static_page(url: str) -> str:
    """
    Get HTML of website using a GET request.
    :param url: url to load
    :return: HTML source
    """
    try:
        response = requests.get(url, timeout=STATIC_PAGE_TIMEOUT)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="url could not be fetched")
    if response.status_code not in range(200, 299):
        raise HTTPException(
            status_code=400,
            detail=f"invalid status code from fetched url: {response.status_code}",
        )
    return response.text


async def load_dynamic_page(url: str) -> str:
    """
    Launch a headless Firefox, navigate to URL, wait some time and return the result HTML.
    :param url: url to load
    :return: HTML after rendering
    """
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get(url)
    time.sleep(DYNAMIC_PAGE_WAIT)
    html = driver.execute_script("return document.body.innerHTML;")
    driver.quit()

    return html


async def load_article(url: str) -> str:
    """
    First try to load an article from static HTML. If no article is found, use selenium to render JS and try again.
    :param url: url of the containing website
    :return: found article text
    """
    static_page = await load_static_page(url)
    static_page_article = scraper.extract_article(static_page)
    if static_page_article:
        return static_page_article

    dynamic_page = await load_dynamic_page(url)
    dynamic_page_article = scraper.extract_article(dynamic_page)
    if dynamic_page_article:
        return dynamic_page_article

    raise HTTPException(status_code=400, detail="could not find article on page")


class ArticleRating(BaseModel):
    article: str
    rating: str
    confidence: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/rate_article", response_class=HTMLResponse)
async def rate_article_form(request: Request, article_text: Annotated[str, Form()]):
    rating, confidence = predict(article_text, use_cuda=USE_CUDA)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "text": article_text,
            "rating": rating,
            "confidence": int(confidence * 100),
        },
    )


@app.post("/rate_url", response_class=HTMLResponse)
async def rate_url_form(request: Request, article_url: Annotated[str, Form()]):
    article_text = await load_article(article_url)
    return rate_article_form(request, article_text)


@app.get("/rate_article")
async def rate_article_text(article_text: str):
    rating, confidence = predict(article_text, use_cuda=USE_CUDA)
    return ArticleRating(article=article_text, rating=rating, confidence=confidence)


@app.get("/rate_url")
async def rate_url(url: str):
    article_text = await load_article(url)
    return rate_article_text(article_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, ssl_keyfile=None, ssl_certfile=None)
