from typing import List, Union
from fastapi import FastAPI, Request, File, UploadFile, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

import os
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)

class PathManager():
   """
   In charge of changing work directory, need to construct it in the main code.
   """
   def __init__(self, rootPath) -> None:
      self.rootPath = rootPath

   def backToRoot(self):
      os.chdir(self.rootPath)

   def createUploadPath(self):
      UPLOAD_ROOT_PATH = "./upload_data"
      try:
         os.mkdir(UPLOAD_ROOT_PATH)
         os.chdir(UPLOAD_ROOT_PATH)
      except:
         os.chdir(UPLOAD_ROOT_PATH)

   def createDirectoryInUploadPath(self, directoryName):
      try:
         shutil.rmtree(directoryName)
         os.mkdir(directoryName)
      except Exception as e: 
         os.mkdir(directoryName)


global pathManager

# back to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")
pathManager = PathManager(os.getcwd())


# Function
@app.get("/")
def read_root():
   return {"Hello": "World", "s":os.path.dirname(os.path.abspath(__file__))}

# developing
@app.get("/pipeline/platform")
def pipeline_platform(request: Request):
   return templates.TemplateResponse("platform.html",{"request":request})

@app.get("/pipeline/data")
def pipeline_data(request: Request):
   return templates.TemplateResponse("data.html",{"request":request})

@app.post("/pipeline/data")
async def pipeline_data_upload(request: Request, directoryName: str = Form(default=None, max_length=50), file_x: UploadFile = File(...), file_y: UploadFile = File(...)):  # directory__Name: Union[str, None] = Query(default=None, max_length=50)
      
      global pathManager

      pathManager.createUploadPath()
      pathManager.createDirectoryInUploadPath(directoryName)
      
      filelist = [file_x, file_y]
      
      for i, file in enumerate(filelist):
         
         try:
            contents = await file.read()
            # X(labeling)
            if i == 0:
               with open(os.path.join(directoryName, "X_" + file.filename), 'wb') as f:
                     f.write(contents)
            # Y(labeling)
            elif i == 1:
               with open(os.path.join(directoryName, "Y_" + file.filename), 'wb') as f:
                     f.write(contents)
         except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
         finally:
            await file.close()

      pathManager.backToRoot()   

      return templates.TemplateResponse("data.html", context = {'request': request, "directoryName": directoryName, "filename_x" : file_x.filename, "filename_y" : file_y.filename})


@app.get("/pipeline/model")
def pipeline_model(request: Request):
   return templates.TemplateResponse("model.html",{"request":request})

@app.get("/pipeline/service")
def pipeline_service(request: Request):
   return templates.TemplateResponse("service.html",{"request":request})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}

