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
import autoPush

# testing
# Append absolute path to import module within different directory.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# print(parent, root)

from model_file import custoNet_s2, custoNet_s1, riro, ribo, custoNet_SLFN
# from model_file.riro import main
from modelParameter import ModelParameter, ModelParameter2, ModelParameter2LayerNet
from apps import evaluating, saving 

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)

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
async def pipeline_data_upload(request: Request, \
                               directoryName: str = Form(default=None, max_length=50), \
                               file_x: UploadFile = File(...), file_y: UploadFile = File(...)):  
                               # directory__Name: Union[str, None] = Query(default=None, max_length=50)
      
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

      return templates.TemplateResponse("data.html", \
                                        context = \
                                       {"request": request, \
                                        "directoryName": directoryName, \
                                        "filename_x" : file_x.filename, \
                                        "filename_y" : file_y.filename})

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
                     matchingTimes: str = Form(default=None, max_length=50), \
                     regularizingStrength: str = Form(default=None, max_length=50), \
                     thresholdForError : str = Form(default=None, max_length=50)):


   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]

   # Get data shape
   dataShape = ModelParameter.get_dataShape(f"{root}\\upload_data\\{dataDirectory}")

   # Define modelParameter
   model_params = ModelParameter(dataDirectory=dataDirectory, \
                                 dataShape=dataShape, \
                                 modelFile=modelFile, \
                                 inputDimension=dataShape["X"][1], \
                                 hiddenNode=1, \
                                 outputDimension=dataShape["Y"][1], \
                                 initializingNumber=eval_avoidNone(initializingNumber), \
                                 lossFunction=lossFunction, \
                                 initializingLearningGoal=eval_avoidNone(learningGoal), \
                                 learningRate=eval_avoidNone(learningRate), \
                                 regularizingLearningRateLowerBound=eval_avoidNone(learningRateLowerBound), \
                                 optimizer=optimizer, \
                                 matchingTimes=eval_avoidNone(matchingTimes), \
                                 regularizingStrength=eval_avoidNone(regularizingStrength), \
                                 thresholdForError=eval_avoidNone(thresholdForError))
   # Train model
   model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

   if model_experiments_record == "Initializing 失敗" or model_experiments_record == "Cramming 失敗":
      training_error_msg = ""

      if model_experiments_record == "Initializing 失敗" : 
         training_error_msg = "Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"
      elif model_experiments_record == "Cramming 失敗" : 
         training_error_msg = "Cramming 失敗，請將超參數 Learning goal 增加"

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
                     "dataShape":dataShape, \
                     "initializingNumber":initializingNumber, \
                     "modelFile":modelFile, \
                     "lossFunction":lossFunction, \
                     "learningGoal":learningGoal, \
                     "learningRate":learningRate, \
                     "learningRateLowerBound":learningRateLowerBound, \
                     "optimizer":optimizer, \
                     "matchingTimes":matchingTimes, \
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

@app.get("/pipeline/model/scenario/SLFN")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   
   return templates.TemplateResponse("model_scenario_SLFN.html",{"request":request, "upload_data":upload_data})

@app.post("/pipeline/model/scenario/SLFN")
def pipeline_model(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     hiddenNode : str = Form(default=None, max_length=50), \
                     activationFunction : str = Form(default=None, max_length=50), \
                     epoch : str = Form(default=None, max_length=50), \
                     batchSize : str = Form(default=None, max_length=50), \
                     learningGoal : str = Form(default=None, max_length=50), \
                     thresholdForError : str = Form(default=None, max_length=50), \
                     lossFunction : str = Form(default=None, max_length=50), \
                     optimizer : str = Form(default=None, max_length=50), \
                     learningRate : str = Form(default=None, max_length=50), \
                     betas : str = Form(default=None, max_length=50), \
                     eps : str = Form(default=None, max_length=50), \
                     weightDecay : str = Form(default=None, max_length=50)):



   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]

   # Get data shape
   dataShape = ModelParameter.get_dataShape(f"{root}\\upload_data\\{dataDirectory}")

   # Define modelParameter
   model_params = ModelParameter2LayerNet(dataDirectory=dataDirectory, \
                                 dataShape=dataShape, \
                                 inputDimension=dataShape["X"][1], \
                                 hiddenNode=eval_avoidNone(hiddenNode), \
                                 outputDimension=dataShape["Y"][1], \
                                 modelFile = "custoNet_SLFN.py", \
                                 activationFunction = activationFunction, \
                                 epoch = eval_avoidNone(epoch), \
                                 batchSize = eval_avoidNone(batchSize), \
                                 learningGoal = eval_avoidNone(learningGoal), \
                                 thresholdForError = eval_avoidNone(thresholdForError), \
                                 lossFunction = lossFunction, \
                                 optimizer = optimizer, \
                                 learningRate = eval_avoidNone(learningRate), \
                                 betas = betas, \
                                 eps = eval_avoidNone(eps), \
                                 weightDecay = eval_avoidNone(weightDecay))

   # Train model
   model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

   if model_experiments_record == "Initializing 失敗" or model_experiments_record == "Cramming 失敗":
      training_error_msg = ""

      if model_experiments_record == "Initializing 失敗" : 
         training_error_msg = "Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"
      elif model_experiments_record == "Cramming 失敗" : 
         training_error_msg = "Cramming 失敗，請將超參數 Learning goal 增加"

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
   
   return templates.TemplateResponse("model_scenario_SLFN.html", \
            context={"request":request, \
                     "upload_data":upload_data, \
                     "dataDirectory":dataDirectory, \
                     "dataShape":dataShape, \
                     "hiddenNode":hiddenNode, \
                     "activationFunction":activationFunction, \
                     "epoch":epoch, \
                     "batchSize":batchSize, \
                     "learningGoal":learningGoal, \
                     "thresholdForError":thresholdForError, \
                     "lossFunction":lossFunction, \
                     "optimizer":optimizer, \
                     "learningRate":learningRate, \
                     "betas":betas, \
                     "eps":eps, \
                     "weightDecay":weightDecay, \
                     "model_experiments_record":model_experiments_record, \
                     # "trainingAccuracy":model_experiments_record["experiments_record"]["train"]["mean_acc"], \
                     # "validatingAccuracy":model_experiments_record["experiments_record"]["valid"]["mean_acc"], \
                     # "url_path_for_Accuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Accuracy.png'), \
                     # "url_path_for_Loss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Loss.png'), \
                     # "url_path_for_Nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Nodes.png'), \
                     # "url_path_for_Pruned_nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Pruned_nodes.png'), \
                     # "url_path_for_Routes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/Routes.png')
                     # 
                     })


@app.get("/pipeline/model/scenario/ASLFN")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   model_file = [pythonFile for pythonFile in os.listdir(f"{root}\\model_file") if pythonFile.endswith(".py")]
   
   return templates.TemplateResponse("model_scenario_ASLFN.html",{"request":request, "upload_data":upload_data, "model_file":model_file})

@app.post("/pipeline/model/scenario/ASLFN")
def pipeline_model(request: Request, \
                     dataDirectory : str = Form(default=None, max_length=50), \
                     dataDescribing : str = Form(default=None, max_length=50), \
                     hiddenNode : str = Form(default=None, max_length=50), \
                     thresholdForError : str = Form(default=None, max_length=50), \
                     activationFunction : str = Form(default=None, max_length=50), \
                     lossFunction : str = Form(default=None, max_length=50), \
                     optimizer : str = Form(default=None, max_length=50), \
                     learningRate : str = Form(default=None, max_length=50), \
                     betas : str = Form(default=None, max_length=50), \
                     eps : str = Form(default=None, max_length=50), \
                     weightDecay : str = Form(default=None, max_length=50), \
                     initializingRule : str = Form(default=None, max_length=50), \
                     initializingNumber : str = Form(default=None, max_length=50), \
                     initializingLearningGoal : str = Form(default=None, max_length=50), \
                     selectingRule : str = Form(default=None, max_length=50), \
                     matchingRule : str = Form(default=None, max_length=50), \
                     matchingTimes : str = Form(default=None, max_length=50), \
                     matchingLearningGoal : str = Form(default=None, max_length=50), \
                     matchingLearningRateLowerBound : str = Form(default=None, max_length=50), \
                     crammingRule : str = Form(default=None, max_length=50), \
                     reorganizingRule : str = Form(default=None, max_length=50), \
                     regularizingTimes : str = Form(default=None, max_length=50), \
                     regularizingStrength : str = Form(default=None, max_length=50), \
                     regularizingLearningGoal : str = Form(default=None, max_length=50), \
                     regularizingLearningRateLowerBound : str = Form(default=None, max_length=50)):

   # List the upload data to Dropdownlist
   upload_data = os.listdir(f"{root}\\upload_data")

   # Get data shape
   dataShape = ModelParameter.get_dataShape(f"{root}\\upload_data\\{dataDirectory}")

   # Define modelParameter
   model_params = ModelParameter2(dataDirectory=dataDirectory, \
                                 dataDescribing=dataDescribing, \
                                 dataShape=dataShape, \
                                 modelFile="custoNet_s2.py", \
                                 inputDimension=dataShape["X"][1], \
                                 hiddenNode=eval_avoidNone(hiddenNode), \
                                 outputDimension=dataShape["Y"][1], \
                                 thresholdForError=eval_avoidNone(thresholdForError), \
                                 activationFunction=activationFunction, \
                                 lossFunction=lossFunction, \
                                 optimizer=optimizer, \
                                 learningRate=eval_avoidNone(learningRate), \
                                 betas=eval_avoidNone(betas), \
                                 eps=eval_avoidNone(eps), \
                                 weightDecay=eval_avoidNone(weightDecay), \
                                 initializingRule=initializingRule, \
                                 initializingNumber=eval_avoidNone(initializingNumber), \
                                 initializingLearningGoal=eval_avoidNone(initializingLearningGoal), \
                                 selectingRule=selectingRule, \
                                 matchingRule=matchingRule, \
                                 matchingTimes=eval_avoidNone(matchingTimes), \
                                 matchingLearningGoal=eval_avoidNone(matchingLearningGoal), \
                                 matchingLearningRateLowerBound=eval_avoidNone(matchingLearningRateLowerBound), \
                                 crammingRule=crammingRule, \
                                 reorganizingRule=reorganizingRule, \
                                 regularizingTimes=eval_avoidNone(regularizingTimes), \
                                 regularizingStrength=eval_avoidNone(regularizingStrength), \
                                 regularizingLearningGoal=eval_avoidNone(regularizingLearningGoal), \
                                 regularizingLearningRateLowerBound=eval_avoidNone(regularizingLearningRateLowerBound))

   # # Train model
   model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

   if model_experiments_record == "Initializing 失敗" or model_experiments_record == "Cramming 失敗":
      training_error_msg = ""

      if model_experiments_record == "Initializing 失敗" : 
         training_error_msg = "Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"
      elif model_experiments_record == "Cramming 失敗" : 
         training_error_msg = "Cramming 失敗，請將超參數 Learning goal 增加"

      return templates.TemplateResponse("model_scenario_1.html", \
            context={"request":request,  \
                     "upload_data":upload_data, \
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
   
   return templates.TemplateResponse("model_scenario_ASLFN.html", \
                                    context={"request":request, \
                                             "upload_data":upload_data, \
                                             "dataDirectory":dataDirectory, \
                                             "dataDescribing":dataDescribing, \
                                             "dataShape":dataShape, \
                                             "hiddenNode":hiddenNode, \
                                             "activationFunction":activationFunction, \
                                             "lossFunction":lossFunction, \
                                             "optimizer":optimizer, \
                                             "learningRate":learningRate, \
                                             "betas":betas, \
                                             "eps":eps, \
                                             "weightDecay":weightDecay, \
                                             "matchingTimes":matchingTimes, \
                                             "initializingRule":initializingRule, \
                                             "initializingNumber":initializingNumber, \
                                             "initializingLearningGoal":initializingLearningGoal, \
                                             "selectingRule":selectingRule, \
                                             "matchingRule":matchingRule, \
                                             "matchingLearningGoal":matchingLearningGoal, \
                                             "matchingLearningRateLowerBound":matchingLearningRateLowerBound, \
                                             "crammingRule":crammingRule, \
                                             "reorganizingRule":reorganizingRule, \
                                             "regularizingTimes":regularizingTimes, \
                                             "regularizingStrength":regularizingStrength, \
                                             "regularizingLearningGoal":regularizingLearningGoal, \
                                             "regularizingLearningRateLowerBound":regularizingLearningRateLowerBound, \
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
   

   model_file_str = model_params.kwargs["modelFile"].split(".")[0] # riro.py -> riro
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

def __model_deploying(modelFile):

   # save model to directory : model_deploying
   saving.writeIntoModelDeploying(modelFile)

   # git add/commit/push automatically
   autoPush.main()

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
                     "learningGoal":model_params.initializingLearningGoal, \
                     "learningRate":model_params.learningRate, \
                     "learningRateLowerBound":model_params.regularizingLearningRateLowerBound, \
                     "optimizer":model_params.optimizer, \
                     "tuningTimes":model_params.matchingTimes, \
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


@app.post("/pipeline/deploy")
def pipeline_deploy(request: Request, \
                     modelFile : str = Form(default=None, max_length=50)):

   __model_deploying(modelFile)

   return modelFile

@app.post("/save/service")
def save_service(model_params, model_perf, model_perf_fig):
   print(f'已進入 save_service()')
   print(f'model_params = {model_params}')
   print(f'model_perf = {model_perf}')
   print(f'model_perf_fig = {model_perf_fig}')

@app.post("/save/model")
def save_model(model_experiments_record=None, model_params=None, model_fig_drt=None):

   saving.writeIntoModelRegistry(model_experiments_record, model_params, model_fig_drt)

def eval_avoidNone(parameter):
   # eval processing, can't using function when input is None
   if parameter is not None:
      return eval(parameter)
   else:
      return None

if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) # 若有 rewrite file 可能不行 reload=True，不然會一直重開 by 李忠大師