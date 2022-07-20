from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/pipeline/platform")
def pipeline_platform(request: Request):
   return templates.TemplateResponse("platform.html",{"request":request})

@app.get("/pipeline/data")
def pipeline_data(request: Request):
   return templates.TemplateResponse("data.html",{"request":request})

@app.get("/pipeline/model")
def pipeline_model(request: Request):
   return templates.TemplateResponse("model.html",{"request":request})

@app.get("/pipeline/service")
def pipeline_service(request: Request):
   return templates.TemplateResponse("service.html",{"request":request})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}