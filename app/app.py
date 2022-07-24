from typing import List
from fastapi import FastAPI, Request, File, UploadFile, Form
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

# testing 
@app.get('/form')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})


@app.post('/form')
def form_post(request: Request, num: int = Form(...)):
    result = num
    print(request)
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result, 'num': num})

# testing 
@app.get('/f/')
async def ff(request:Request):
    return templates.TemplateResponse('f1.html',{"request":request})

@app.post('/upfile/')
async def up_f(request:Request,file_list:List[bytes]=File(...)):
    return templates.TemplateResponse('f.html',{"request":request,"file_sizes":[len(dd)/1024 for dd in file_list]})

@app.post('/upfile1/')
async def up_f1(request:Request,file_list:List[UploadFile]=File(...)):
    return templates.TemplateResponse('f.html',{"request":request,"file_names":[dd.filename for dd in file_list]})



# developing
@app.get("/pipeline/platform")
def pipeline_platform(request: Request):
   return templates.TemplateResponse("platform.html",{"request":request})

@app.get("/pipeline/data")
def pipeline_data(request: Request):
   return templates.TemplateResponse("data.html",{"request":request})

@app.post("/pipeline/data/upload")
async def pipeline_data_upload_x(request: Request, file_x: UploadFile = File(...), file_y: UploadFile = File(...)): 
      filelist = [file_x, file_y]
      for file in filelist:
         try:
            contents = await file.read()
            with open(file.filename, 'wb') as f:
                  f.write(contents)
         except Exception:
            return {"message": "There was an error uploading the file"}
         finally:
            await file.close()
      return templates.TemplateResponse("data.html", context = {'request': request, "filename_x" : file_x.filename, "filename_y" : file_y.filename})


@app.get("/pipeline/model")
def pipeline_model(request: Request):
   return templates.TemplateResponse("model.html",{"request":request})

@app.get("/pipeline/service")
def pipeline_service(request: Request):
   return templates.TemplateResponse("service.html",{"request":request})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}