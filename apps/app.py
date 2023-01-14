from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import os
import requests
import time
import autoPush
import autoStartContainer
import numpy as np
import pandas as pd

# testing
# Append absolute path to import module within different directory.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from model_file import ASLFN, SLFN
from modelParameter import ModelParameter
from apps import evaluating, saving 
from ymlEditing import deployingModelToYml, revokingModelToYml

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)

autoStartContainer.main()

# Function
@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/entry")
def entry(request: Request):
   return templates.TemplateResponse("entry.html",{"request":request})

@app.get("/pipeline/develop")
def pipeline_platform(request: Request):
   return templates.TemplateResponse("develop.html",{"request":request})

@app.get("/pipeline/platform")
def pipeline_platform(request: Request):
   return templates.TemplateResponse("platform.html",{"request":request})

@app.get("/pipeline/data")
def pipeline_data(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")

   return templates.TemplateResponse("data.html",{"request":request, "upload_data":upload_data})

@app.post("/pipeline/data")
async def pipeline_data_upload(request: Request, \
                               dataUse: str = Form(default=None, max_length=50), \
                               newDirectory: str = Form(default=None, max_length=50), \
                               existedDirectory: str = Form(default=None, max_length=50), \
                               file_x: UploadFile = File(...), file_y: UploadFile = File(...)):  

      upload_data = os.listdir(f"{root}\\upload_data")

### developing

      # Data validation
      async def dataValidating():
         # print(file_x, file_y)
         filename = file_x.filename
         contents = await file_x.read()
         print(filename, contents)

         filename = file_y.filename
         contents = await file_y.read()
         print(filename, contents)


      dataValidating()

### 

      # Data uploading and definition 
      if dataUse == "Train" : 

         drtPath = Path(f"{root}\\upload_data\\{newDirectory}\\Train")
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

      elif dataUse == "Test" : 

         drtPath = Path(f"{root}\\upload_data\\{existedDirectory}\\Test")
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

  
      upload_data = os.listdir(f"{root}\\upload_data")
      directory = newDirectory if dataUse == "Train" else existedDirectory

      return templates.TemplateResponse("data.html", \
                                        context = \
                                       {"request": request, \
                                        "upload_data":upload_data, \
                                        "directory": directory, \
                                        "filename_x" : file_x.filename, \
                                        "filename_y" : file_y.filename})

@app.get("/pipeline/model")
def pipeline_model(request: Request):

   return templates.TemplateResponse("model.html",{"request":request})

@app.get("/pipeline/model/scenario/SLFN")
def pipeline_model(request: Request):

   upload_data = os.listdir(f"{root}\\upload_data")
   
   return templates.TemplateResponse("model_scenario_SLFN.html",{"request":request, "upload_data":upload_data})

@app.post("/pipeline/model/scenario/SLFN")
def pipeline_model(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     hiddenNode : str = Form(default=None, max_length=50), \
                     weightInitialization : str = Form(default=None, max_length=50), \
                     activationFunction : str = Form(default=None, max_length=50), \
                     epoch : str = Form(default=None, max_length=50), \
                     batchSize : str = Form(default=None, max_length=50), \
                     learningGoal : str = Form(default=None, max_length=50), \
                     testingMetric : str = Form(default=None, max_length=50), \
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
   model_params = ModelParameter(dataDirectory=dataDirectory, \
                                 dataShape=dataShape, \
                                 inputDimension=dataShape["X"][1], \
                                 hiddenNode=eval_avoidNone(hiddenNode), \
                                 outputDimension=dataShape["Y"][1], \
                                 modelFile = "SLFN.py", \
                                 weightInitialization = weightInitialization, \
                                 activationFunction = activationFunction, \
                                 epoch = eval_avoidNone(epoch), \
                                 batchSize = eval_avoidNone(batchSize), \
                                 learningGoal = eval_avoidNone(learningGoal), \
                                 testingMetric = testingMetric, \
                                 lossFunction = lossFunction, \
                                 optimizer = optimizer, \
                                 learningRate = eval_avoidNone(learningRate), \
                                 betas = eval_avoidNone(betas), \
                                 eps = eval_avoidNone(eps), \
                                 weightDecay = eval_avoidNone(weightDecay))

   # Train model
   network, model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

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
   save_model(network, model_experiments_record, model_params, model_fig_drt)

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
                     "weightInitialization":weightInitialization, \
                     "activationFunction":activationFunction, \
                     "epoch":epoch, \
                     "batchSize":batchSize, \
                     "learningGoal":learningGoal, \
                     "testingMetric":testingMetric, \
                     "lossFunction":lossFunction, \
                     "optimizer":optimizer, \
                     "learningRate":learningRate, \
                     "betas":betas, \
                     "eps":eps, \
                     "weightDecay":weightDecay, \
                     "model_experiments_record":model_experiments_record, \
                     "trainingAccuracy":model_experiments_record["experiments_record"]["train"]["mean_acc"], \
                     "validatingAccuracy":model_experiments_record["experiments_record"]["valid"]["mean_acc"], \
                     "url_path_for_trainingAccuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingAccuracy.png'), \
                     "url_path_for_trainingLoss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingLoss.png'), \
                     "url_path_for_validatingAccuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/validatingAccuracy.png'), \
                     "url_path_for_validatingLoss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/validatingLoss.png'), \
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
                     weightInitialization : str = Form(default=None, max_length=50), \
                     activationFunction : str = Form(default=None, max_length=50), \
                     testingMetric : str = Form(default=None, max_length=50), \
                     lossFunction : str = Form(default=None, max_length=50), \
                     optimizer : str = Form(default=None, max_length=50), \
                     learningRate : str = Form(default=None, max_length=50), \
                     betas : str = Form(default=None, max_length=50), \
                     eps : str = Form(default=None, max_length=50), \
                     weightDecay : str = Form(default=None, max_length=50), \
                     initializingRule : str = Form(default=None, max_length=50), \
                     initializingNumber : str = Form(default=None, max_length=50), \
                     learningGoal : str = Form(default=None, max_length=50), \
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
   model_params = ModelParameter(dataDirectory=dataDirectory, \
                                 dataDescribing=dataDescribing, \
                                 dataShape=dataShape, \
                                 modelFile="ASLFN.py", \
                                 inputDimension=dataShape["X"][1], \
                                 hiddenNode=eval_avoidNone(hiddenNode), \
                                 outputDimension=dataShape["Y"][1], \
                                 weightInitialization=weightInitialization, \
                                 activationFunction=activationFunction, \
                                 testingMetric=testingMetric, \
                                 lossFunction=lossFunction, \
                                 optimizer=optimizer, \
                                 learningRate=eval_avoidNone(learningRate), \
                                 betas=eval_avoidNone(betas), \
                                 eps=eval_avoidNone(eps), \
                                 weightDecay=eval_avoidNone(weightDecay), \
                                 initializingRule=initializingRule, \
                                 initializingNumber=eval_avoidNone(initializingNumber), \
                                 learningGoal=eval_avoidNone(learningGoal), \
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

   # Train model
   network, model_experiments_record, model_params, model_fig_drt = __model_training(model_params)

   if model_experiments_record == "Initializing 失敗" or model_experiments_record == "Cramming 失敗":
      training_error_msg = ""

      if model_experiments_record == "Initializing 失敗" : 
         training_error_msg = "Initializing 失敗，請將超參數 Initializing number 減少，或是將超參數 Learning goal 增加"
      elif model_experiments_record == "Cramming 失敗" : 
         training_error_msg = "Cramming 失敗，請將超參數 Learning goal 增加"

      return templates.TemplateResponse("model_scenario_ASLFN.html", \
            context={"request":request,  \
                     "upload_data":upload_data, \
                     "interrupted_message":training_error_msg})

   # Save model config a& Perf.
   save_model(network, model_experiments_record, model_params, model_fig_drt)

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
                                             "weightInitialization":weightInitialization, \
                                             "activationFunction":activationFunction, \
                                             "testingMetric":testingMetric, \
                                             "lossFunction":lossFunction, \
                                             "optimizer":optimizer, \
                                             "learningRate":learningRate, \
                                             "betas":betas, \
                                             "eps":eps, \
                                             "weightDecay":weightDecay, \
                                             "matchingTimes":matchingTimes, \
                                             "initializingRule":initializingRule, \
                                             "initializingNumber":initializingNumber, \
                                             "learningGoal":learningGoal, \
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
                                             "url_path_for_trainingAccuracy":app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingAccuracy.png'), \
                                             "url_path_for_trainingLoss":app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingLoss.png'), \
                                             "url_path_for_Nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/nodes.png'), \
                                             "url_path_for_Pruned_nodes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/prunedNodes.png'), \
                                             "url_path_for_Routes":app.url_path_for('model_fig', path=f'/{model_fig_drt}/routes.png')
                                             
                                             })

def __model_training(model_params):
   

   model_file_str = model_params.kwargs["modelFile"].split(".")[0] # riro.py -> riro
   model = eval(model_file_str) # str to module through eval function : riro, ribo, biro, bibo
   network, model_experiments_record, model_params, model_fig_drt = model.main(model_params)

   return network, model_experiments_record, model_params, model_fig_drt

# call container service to get rmseError 
def __model_evaluating(dataDirectory, modelFile):
   
   x_test, y_test = evaluating.reading_dataset_Testing(dataDirectory)
   checkpoints = evaluating.reading_pkl(modelFile)
   network = evaluating.reading_pt(modelFile)

   # network = checkpoints["model_experiments_record"]["network"]
   model_experiments_record = checkpoints["model_experiments_record"]
   model_params = checkpoints["model_params"]
   model_fig_drt = checkpoints["model_fig_drt"]
   
   # rmseError 應該不能跟原始pkl檔存一起，要另外存
   # rmseError = evaluating.inferencing(network, x_test, y_test)

   return model_experiments_record, model_params, model_fig_drt

# call local pkl to predict and get rmseError
def __model_evaluating_old_service(dataDirectory, modelFile):
   
   x_test, y_test = evaluating.reading_dataset_Testing(dataDirectory)
   checkpoints = evaluating.reading_pkl(modelFile)

   network = checkpoints["model_experiments_record"]["network"]
   model_experiments_record = checkpoints["model_experiments_record"]
   model_params = checkpoints["model_params"]
   model_fig_drt = checkpoints["model_fig_drt"]
   
   # rmseError 應該不能跟原始pkl檔存一起，要另外存
   rmseError = evaluating.inferencing(network, x_test, y_test)

   return model_experiments_record, model_params, model_fig_drt, rmseError

def __model_deploying(modelId):

   # mongoDB import(worked)
   modelPklFile = requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelId&value={modelId}").json()[0]["modelName"]
   # save model to directory : model_deploying
   modelPtFile = modelPklFile[:-3] + "pt"
   saving.writeIntoDockerApps(modelPtFile)

   # find Available Port and write into .gitlab-ci.yml file
   deployingModelToYml(modelId)

   # 感覺會有因為沒有檔案變動而無法 commit 的狀況，可能要寫個額外的檔案變動，避免錯誤
   with open(f"{root}\\ASLFN\\docker_apps\\deployTmp", "w", encoding="utf-8") as file:
      file.write(f"避免沒有其他檔案更動而生成的檔案 : {time.time()}")

   # git add/commit/push automatically
   autoPush.main()

def __model_revoking(modelId):

   # find containerID by modelId and write into .gitlab-ci.yml file
   revokingModelToYml(modelId)

   # 感覺會有因為沒有檔案變動而無法 commit 的狀況，可能要寫個額外的檔案變動，避免錯誤
   with open(f"{root}\\apps\\revokeTmp", "w", encoding="utf-8") as file:
      file.write(f"避免沒有其他檔案更動而生成的檔案 : {time.time()}")
   
   # git add/commit/push automatically
   autoPush.main()

# new /service
@app.get("/pipeline/service")
def pipeline_service(request: Request):

   # mongoDB import(worked)
   key = "deployStatus"
   value = "deploying"
   predictAPI_lst = [data["modelName"] for data in requests.get(f"http://127.0.0.1:8001/model/deployments?key={key}&value={value}").json()]

   return templates.TemplateResponse("service.html",{"request":request, "predictAPI_lst":predictAPI_lst})

@app.post("/pipeline/service")
def pipeline_service(request: Request, \
                     predictAPI: str = Form(default=None, max_length=50)):

   # GET
   # mongoDB import(worked)
   key = "deployStatus"
   value = "deploying"
   predictAPI_lst = [data["modelName"] for data in requests.get(f"http://127.0.0.1:8001/model/deployments?key={key}&value={value}").json()]

   # POST
   # mongoDB import
   modelId = requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelName&value={predictAPI}").json()[0]["modelId"]
   dataDirectory = requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelName&value={predictAPI}").json()[0]["trainedDataset"]
   modelPklFile = predictAPI
   servicePort = requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelName&value={predictAPI}").json()[0]["containerPort"]
   
   x_test, y_test = evaluating.reading_dataset_Testing(dataDirectory) 
   rawTestingData = {"x_test" : x_test.tolist(), "y_test" : y_test.tolist()}

   # a = time.time()
   res = requests.post(f"http://127.0.0.1:{servicePort}/predict", json={"dataDirectory": rawTestingData})
   rmseError = res.json()["rmseError"]
   # print(f"MongoDB : {time.time()-a}")

   model_experiments_record, model_params, model_fig_drt = __model_evaluating(dataDirectory, modelPklFile)

   app.mount(
      f"/model_fig",
      StaticFiles(directory=Path(__file__).parent.parent.absolute() / "model_fig"), #  / img_drt
      name="model_fig",
   )  

   if "ASLFN" in modelPklFile:
      url_path_for_fig_1 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingAccuracy.png')
      url_path_for_fig_2 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingLoss.png')
      url_path_for_fig_3 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/nodes.png')
      url_path_for_fig_4 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/prunedNodes.png')
      url_path_for_fig_5 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/routes.png')

   else:
      url_path_for_fig_1 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingAccuracy.png')
      url_path_for_fig_2 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/trainingLoss.png')
      url_path_for_fig_3 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/validatingAccuracy.png')
      url_path_for_fig_4 = app.url_path_for('model_fig', path=f'/{model_fig_drt}/validatingLoss.png')
      url_path_for_fig_5 = None



   return templates.TemplateResponse("service.html", \
                  context={"request":request, \
                     "predictAPI_lst":predictAPI_lst, \
                     "dataDirectory":dataDirectory, \
                     "dataShape":model_params.kwargs["dataShape"], \
                     "modelPklFile":modelPklFile, \
                     "hiddenNode":model_params.kwargs["hiddenNode"], \
                     "weightInitialization":model_params.kwargs["weightInitialization"], \
                     "activationFunction":model_params.kwargs["activationFunction"], \
                     "epoch":template_avoidNone(model_params, "epoch"), \
                     "batchSize":template_avoidNone(model_params, "batchSize"), \
                     "lossFunction":model_params.kwargs["lossFunction"], \
                     "optimizer":model_params.kwargs["optimizer"], \
                     "learningRate":model_params.kwargs["learningRate"] if model_params.kwargs["learningRate"] else None, \
                     "betas":model_params.kwargs["betas"], \
                     "eps":model_params.kwargs["eps"], \
                     "weightDecay":model_params.kwargs["weightDecay"], \
                     "initializingRule":template_avoidNone(model_params, "initializingRule"), \
                     "initializingNumber":template_avoidNone(model_params, "initializingNumber"), \
                     "learningGoal":model_params.kwargs["learningGoal"], \
                     "selectingRule":template_avoidNone(model_params, "selectingRule"), \
                     "matchingRule":template_avoidNone(model_params, "matchingRule"), \
                     "matchingTimes":template_avoidNone(model_params, "matchingTimes"), \
                     "matchingLearningGoal":template_avoidNone(model_params, "matchingLearningGoal"), \
                     "matchingLearningRateLowerBound":template_avoidNone(model_params, "matchingLearningRateLowerBound"), \
                     "crammingRule":template_avoidNone(model_params, "crammingRule"), \
                     "reorganizingRule":template_avoidNone(model_params, "reorganizingRule"), \
                     "regularizingTimes":template_avoidNone(model_params, "regularizingTimes"), \
                     "regularizingStrength":template_avoidNone(model_params, "regularizingStrength"), \
                     "regularizingLearningGoal":template_avoidNone(model_params, "regularizingLearningGoal"), \
                     "regularizingLearningRateLowerBound":template_avoidNone(model_params, "regularizingLearningRateLowerBound"), \
                     "trainingAccuracy":model_experiments_record["experiments_record"]["train"]["mean_acc"], \
                     "validatingAccuracy":model_experiments_record["experiments_record"]["valid"]["mean_acc"], \
                     "rmseError":rmseError, \
                     "url_path_for_fig_1":url_path_for_fig_1, \
                     "url_path_for_fig_2":url_path_for_fig_2, \
                     "url_path_for_fig_3":url_path_for_fig_3, \
                     "url_path_for_fig_4":url_path_for_fig_4, \
                     "url_path_for_fig_5":url_path_for_fig_5, \
                     })


@app.get("/pipeline/deploy")
def pipeline_service(request: Request):
   
   # mongoDB import(worked)
   deployRecord = requests.get(f"http://127.0.0.1:8001/model/deployments/all").json()


   return templates.TemplateResponse("deploy.html", \
                  context={"request":request, \
                           "deployRecord":deployRecord, \
                           })


@app.post("/pipeline/deploy")
def pipeline_deploy(request: Request, \
                     modelId: int = Form(default=None), \
                     deployStatus: str = Form(default=None, max_length=50)):


   changingStatus(modelId)
   # mongoDB import
   deployRecord = requests.get(f"http://127.0.0.1:8001/model/deployments/all").json()


   if deployStatus == "deploying":
      __model_deploying(modelId)
   elif deployStatus == "revoking":
      __model_revoking(modelId)

   return templates.TemplateResponse("deploy.html", \
               context={"request":request, \
                        "deployRecord":deployRecord, \
                        })


@app.get("/post_8002/{data}")
def request_8002(data:str):
   
   # reading_dataset_Testing 要做個調整，先對 training data 產生一個 sc，再去對 testing data 做 transform
   x_test, y_test = evaluating.reading_dataset_Testing(data) 
   # rawTestingData = {"a" : np.array([1,2,3]).tolist()}
   rawTestingData = {"x_test" : x_test.tolist(), "y_test" : y_test.tolist()}
   # modelPklFile = "solar_ASLFN_0.6_0.5_221007_101128.pkl"
   res = requests.post("http://127.0.0.1:8360/predict", json={"dataDirectory": rawTestingData})
   return res.json()



@app.post("/save/service")
def save_service(model_params, model_perf, model_perf_fig):
   print(f'已進入 save_service()')
   print(f'model_params = {model_params}')
   print(f'model_perf = {model_perf}')
   print(f'model_perf_fig = {model_perf_fig}')

@app.post("/save/model")
def save_model(network=None, model_experiments_record=None, model_params=None, model_fig_drt=None):

   saving.writeIntoModelRegistry(network, model_experiments_record, model_params, model_fig_drt)

def eval_avoidNone(parameter):
   # eval processing, can't using function when input is None
   if parameter is not None:
      return eval(parameter)
   else:
      return None

def template_avoidNone(model_params, keys):
   if keys not in model_params.kwargs.keys():
      return None
   else:
      return model_params.kwargs[keys]


def changingStatus(modelId):

   # mongoDB import
   if requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelId&value={modelId}").json()[0]["deployStatus"] == "deploying":

      deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                                    json={"modelId" : modelId, \
                                          "keyToBeChanged" : "deployStatus", \
                                          "valueToBeChanged" : "revoking"}).json()

   elif requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelId&value={modelId}").json()[0]["deployStatus"] == "revoking":

      deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                                    json={"modelId" : modelId, \
                                          "keyToBeChanged" : "deployStatus", \
                                          "valueToBeChanged" : "deploying"}).json()

   return deployRecord


if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8000) # 若有 rewrite file 可能不行 reload=True，不然會一直重開 by 李忠大師