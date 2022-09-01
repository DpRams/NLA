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
# app.mount(
#     "/pipeline",
#     StaticFiles(directory=Path(__file__).parent.parent.absolute() / "templates"),
#     name="pipeline",
# )


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
      
      drtPath = Path(f"{root}\\upload_data\\{directoryName}")
      drtPath.mkdir(parents=True, exist_ok=True)
      
      filelist = [file_x, file_y]
      
      for i, file in enumerate(filelist):
         
         try:
            contents = await file.read()
            # X(labeling)
            if i == 0:
               with open(os.path.join(drtPath, "X_" + file.filename), 'wb') as f:
                     f.write(contents)
            # Y(labeling)
            elif i == 1:
               with open(os.path.join(drtPath, "Y_" + file.filename), 'wb') as f:
                     f.write(contents)
         except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
         finally:
            await file.close()

      # pathManager.backToRoot()   

      return templates.TemplateResponse("data.html", context = {'request': request, "directoryName": directoryName, "filename_x" : file_x.filename, "filename_y" : file_y.filename})

@app.get("/pipeline/model")
def pipeline_model(request: Request):

   return templates.TemplateResponse("model.html",{"request":request})

@app.get("/pipeline/model/scenario/1")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]
   
   return templates.TemplateResponse("model_scenario_1.html",{"request":request, "upload_data":upload_data, "model_file":model_file})

@app.post("/pipeline/model/scenario/1")
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


   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]

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

   if model_experiments_record == "Initializing 失敗" or model_experiments_record == "Cramming 失敗":
      training_error_msg = ""

      if model_experiments_record == "Initializing 失敗" : training_error_msg = "Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"
      elif model_experiments_record == "Cramming 失敗" : training_error_msg = "Cramming 失敗，請將超參數 Learning goal 增加"

      return templates.TemplateResponse("model_scenario_1.html", \
            context={"request":request,  \
                     "upload_data":upload_data, \
                     "model_file":model_file, \
                     "interrupted_message":training_error_msg})

   # Save model config a& Perf.
   save_model(model_experiments_record, model_params, model_fig_drt)

   print(f"model_fig_drt = {model_fig_drt}")
   app.mount(
      f"/model_fig",
      StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig"), #  / img_drt
      name="model_fig",
   )  
   print(app.url_path_for('model_fig', path=f'/{model_fig_drt}/Accuracy.png'))
   
   return templates.TemplateResponse("model_scenario_1.html", \
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


@app.get("/pipeline/model/scenario/2")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]
   
   return templates.TemplateResponse("model_scenario_2.html",{"request":request, "upload_data":upload_data, "model_file":model_file})

@app.post("/pipeline/model/scenario/2")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]
   
   # return templates.TemplateResponse("model_scenario_2.html",{"request":request, "upload_data":upload_data, "model_file":model_file})
   return "/pipeline/model/scenario/2 收到 POST"

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

   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")
   model_registry = os.listdir(f"{root}\\model_registry")
   
   return templates.TemplateResponse("service.html",{"request":request, "upload_data":upload_data, "model_registry":model_registry})

@app.post("/pipeline/service")
def pipeline_service(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelFile: str = Form(default=None, max_length=50)):

   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")
   model_registry = os.listdir(f"{root}\\model_registry")

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


if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)