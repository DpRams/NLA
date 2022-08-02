from typing import List, Union
from fastapi import FastAPI, Request, File, UploadFile, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import os
import shutil
import requests
import time

# Append absolute path to import module within different directory.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# print(parent, root)

from model_file import riro, ribo
# from model_file.riro import main
from modelParameter import ModelParameter

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
# parent, root = file.parent, file.parents[1]
pathManager = PathManager(root)


# Function
@app.get("/")
def read_root():
   return {"Hello": "World"}

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

   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_file = [pythonFile for pythonFile in os.listdir("./model_file") if pythonFile.endswith(".py")]
   
   return templates.TemplateResponse("model.html",{"request":request, "upload_data":upload_data, "model_file":model_file})

@app.post("/pipeline/model")
def pipeline_model(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelFile: str = Form(default=None, max_length=50), \
                     lossFunction: str = Form(default=None, max_length=50), \
                     learningGoal: str = Form(default=None, max_length=50), \
                     learningRate: str = Form(default=None, max_length=50), \
                     learningRateUpperBound: str = Form(default=None, max_length=50), \
                     learningRateLowerBound: str = Form(default=None, max_length=50), \
                     optimizer: str = Form(default=None, max_length=50), \
                     regularizingStrength: str = Form(default=None, max_length=50)):

   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_file = [pythonFile for pythonFile in os.listdir("./model_file") if pythonFile.endswith(".py")]

   # Define modelParameter
   model_params = ModelParameter(dataDirectory=dataDirectory, \
                                 modelFile=modelFile, \
                                 lossFunction=lossFunction, \
                                 learningGoal=learningGoal, \
                                 learningRate=learningRate, \
                                 learningRateUpperBound=learningRateUpperBound,\
                                 learningRateLowerBound=learningRateLowerBound, \
                                 optimizer=optimizer, \
                                 regularizingStrength=regularizingStrength)
   # Train model
   model_perf, model_perf_fig = __model_training(model_params)
   # Save model config a& Perf.
   save_model(model_params, model_perf, model_perf_fig)



   return templates.TemplateResponse("model.html", \
            context={"request":request, \
                     "upload_data":upload_data, \
                     "model_file":model_file, \
                     "dataDirectory":dataDirectory, \
                     "modelFile":modelFile, \
                     "lossFunction":lossFunction, \
                     "learningGoal":learningGoal, \
                     "learningRate":learningRate, \
                     "learningRateUpperBound":learningRateUpperBound, \
                     "learningRateLowerBound":learningRateLowerBound, \
                     "optimizer":optimizer, \
                     "regularizingStrength":regularizingStrength})


def __model_training(model_params):
   
   
   print(model_params.info())
   model_file_str = model_params.modelFile.split(".")[0] # riro.py -> riro
   model = eval(model_file_str) # str to module through eval function : riro, ribo, biro, bibo
   model.main()

   model_perf = None
   model_perf_fig = None

   return (model_perf, model_perf_fig)

@app.get("/pipeline/service")
def pipeline_service(request: Request):
   
   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_repo = os.listdir("./model_repo")
   
   return templates.TemplateResponse("service.html",{"request":request, "upload_data":upload_data, "model_repo":model_repo})

@app.post("/pipeline/service")
def pipeline_service(request: Request):

   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_repo = os.listdir("./model_repo")

   return templates.TemplateResponse("service.html",{"request":request, "upload_data":upload_data, "model_repo":model_repo})

@app.post("/save/service")
def save_service(model_params, model_perf, model_perf_fig):
   print(f'已進入 save_service()')
   print(f'model_params = {model_params}')
   print(f'model_perf = {model_perf}')
   print(f'model_perf_fig = {model_perf_fig}')

@app.post("/save/model")
def save_model(model_params, model_perf, model_perf_fig):
   print(f'已進入 save_model()')
   print(f'model_params = {model_params}')
   print(f'model_perf = {model_perf}')
   print(f'model_perf_fig = {model_perf_fig}')


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}



if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)