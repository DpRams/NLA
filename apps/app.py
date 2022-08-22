from typing import List, Union
from fastapi import FastAPI, Request, File, UploadFile, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import os
import shutil
import requests
import time
import urllib.parse

# testing
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
from apps import evaluating, saving 

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

   # template_drt = "__template_drt"
   
   # app.mount(
   #    f"/model_fig/{template_drt}",
   #    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig" / template_drt), 
   #    name="model_fig",
   # )  
   
   return templates.TemplateResponse("model.html",{"request":request, "upload_data":upload_data, "model_file":model_file})

@app.post("/pipeline/model")
def pipeline_model(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelFile: str = Form(default=None, max_length=50), \
                     initializingNumber: str = Form(default=None, max_length=50), \
                     lossFunction: str = Form(default=None, max_length=50), \
                     learningGoal: str = Form(default=None, max_length=50), \
                     learningRate: str = Form(default=None, max_length=50), \
                     learningRateLowerBound: str = Form(default=None, max_length=50), \
                     optimizer: str = Form(default=None, max_length=50), \
                     tuningTimes: str = Form(default=None, max_length=50), \
                     regularizingStrength: str = Form(default=None, max_length=50)):

   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_file = [pythonFile for pythonFile in os.listdir("./model_file") if pythonFile.endswith(".py")]

   # Set default "RMSE", "Adam" for lossFn, optim  
   lossFunction = "RMSE"
   optimizer = "Adam"

   # Define modelParameter
   model_params = ModelParameter(dataDirectory=dataDirectory, \
                                 modelFile=modelFile, \
                                 initializingNumber=int(initializingNumber), \
                                 lossFunction=lossFunction, \
                                 learningGoal=learningGoal, \
                                 learningRate=learningRate, \
                                 learningRateLowerBound=learningRateLowerBound, \
                                 optimizer=optimizer, \
                                 tuningTimes=tuningTimes, \
                                 regularizingStrength=regularizingStrength)
   # Train model
   model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

   if model_experiments_record == "Initializing 失敗":
      return templates.TemplateResponse("model.html", \
            context={"request":request,  \
                     "upload_data":upload_data, \
                     "model_file":model_file, \
                     "interrupted_message":"Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"})

   # Save model config a& Perf.
   save_model(model_experiments_record, model_params, model_fig_drt)

   print(f"model_fig_drt = {model_fig_drt}")
   app.mount(
      f"/model_fig",
      StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig"), #  / img_drt
      name="model_fig",
   )  
   print(app.url_path_for('model_fig', path=f'/{model_fig_drt}/Accuracy.png'))
   
   return templates.TemplateResponse("model.html", \
            context={"request":request, \
                     "upload_data":upload_data, \
                     "model_file":model_file, \
                     "dataDirectory":dataDirectory, \
                     "initializingNumber":initializingNumber, \
                     "modelFile":modelFile, \
                     "lossFunction":lossFunction, \
                     "learningGoal":learningGoal, \
                     "learningRate":learningRate, \
                     "learningRateLowerBound":learningRateLowerBound, \
                     "optimizer":optimizer, \
                     "tuningTimes":tuningTimes, \
                     "regularizingStrength":regularizingStrength, \
                     "model_experiments_record":model_experiments_record, \
                     "trainingAccuracy":model_experiments_record["experiments_record"]["train"]["mean_acc"], \
                     "validatingAccuracy":model_experiments_record["experiments_record"]["valid"]["mean_acc"], \
                     "url_path_for_Accuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Accuracy.png'), \
                     "url_path_for_Loss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Loss.png'), \
                     "url_path_for_Nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Nodes.png'), \
                     "url_path_for_Pruned_nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Pruned_nodes.png'), \
                     "url_path_for_Routes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Routes.png')
                     })


def __model_training(model_params):
   

   model_file_str = model_params.modelFile.split(".")[0] # riro.py -> riro
   model = eval(model_file_str) # str to module through eval function : riro, ribo, biro, bibo
   model_experiments_record, model_params, model_fig_drt = model.main(model_params)

   return model_experiments_record, model_params, model_fig_drt

def __model_evaluating(dataDirectory, modelFile):
   
   x_test, y_test = evaluating.reading_dataset_Testing(dataDirectory)
   checkpoints = evaluating.reading_pkl(modelFile)

   network = checkpoints["model_experiments_record"]["network"]
   model_experiments_record = checkpoints["model_experiments_record"]
   model_params = checkpoints["model_params"]
   model_fig_drt = checkpoints["model_fig_drt"]
   
   # testAccuracy 應該不能跟原始pkl檔存一起，要另外存
   testingAccuracy = evaluating.inferencing(network, x_test, y_test)

   return model_experiments_record, model_params, model_fig_drt, testingAccuracy

@app.get("/pipeline/service")
def pipeline_service(request: Request):
   
   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_registry = os.listdir("./model_registry")

   # template_drt = "__template_drt"
   
   # app.mount(
   #    f"/model_fig/{template_drt}",
   #    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig" / template_drt), 
   #    name="model_fig",
   # )  
   
   return templates.TemplateResponse("service.html",{"request":request, "upload_data":upload_data, "model_registry":model_registry})

@app.post("/pipeline/service")
def pipeline_service(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelFile: str = Form(default=None, max_length=50)):

   global pathManager

   # List the upload data to Dropdownlist
   pathManager.backToRoot()
   upload_data = os.listdir("./upload_data")
   model_registry = os.listdir("./model_registry")

   model_experiments_record, model_params, model_fig_drt, testingAccuracy = __model_evaluating(dataDirectory, modelFile)

   print(f"model_fig_drt = {model_fig_drt}")
   app.mount(
      f"/model_fig",
      StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig"), #  / img_drt
      name="model_fig",
   )  

   return templates.TemplateResponse("service.html", \
                  context={"request":request, \
                     "upload_data":upload_data, \
                     "dataDirectory":dataDirectory, \
                     "modelFile":modelFile, \
                     "initializingNumber":model_params.initializingNumber, \
                     "lossFunction":model_params.lossFunction, \
                     "learningGoal":model_params.learningGoal, \
                     "learningRate":model_params.learningRate, \
                     "learningRateLowerBound":model_params.learningRateLowerBound, \
                     "optimizer":model_params.optimizer, \
                     "tuningTimes":model_params.tuningTimes, \
                     "regularizingStrength":model_params.regularizingStrength, \
                     "model_registry":model_registry, \
                     "trainingAccuracy":model_experiments_record["experiments_record"]["train"]["mean_acc"], \
                     "validatingAccuracy":model_experiments_record["experiments_record"]["valid"]["mean_acc"], \
                     "testingAccuracy":testingAccuracy, \
                     "url_path_for_Accuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Accuracy.png'), \
                     "url_path_for_Loss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Loss.png'), \
                     "url_path_for_Nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Nodes.png'), \
                     "url_path_for_Pruned_nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Pruned_nodes.png'), \
                     "url_path_for_Routes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Routes.png')
                     })


@app.post("/save/service")
def save_service(model_params, model_perf, model_perf_fig):
   print(f'已進入 save_service()')
   print(f'model_params = {model_params}')
   print(f'model_perf = {model_perf}')
   print(f'model_perf_fig = {model_perf_fig}')

@app.post("/save/model")
def save_model(model_experiments_record=None, model_params=None, model_fig_drt=None):

   saving.writeIntoModelRegistry(model_experiments_record, model_params, model_fig_drt)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}


if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)