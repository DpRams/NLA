import time
import pickle
import torch
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import shutil
import json
from network.net import Network


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

def writeIntoModelRegistry(network, model_experiments_record, model_params, model_fig_drt):

    data_drt = model_params.kwargs["dataDirectory"]
    
    if "initializingLearningGoal" in model_params.kwargs.keys():
        lr_goal = model_params.kwargs["initializingLearningGoal"]
    else:
        lr_goal = model_params.kwargs["learningGoal"]

    # create dir  
    pklPath = Path(f"{root}\\model_registry\\pkl\\")
    pklPath.mkdir(parents=True, exist_ok=True)

    timestamp = model_params.kwargs["timestamp"]
    modelType = model_params.kwargs["modelFile"][:-3]
    validAcc = str(model_experiments_record["experiments_record"]["valid"]["mean_acc"])[:5]
    fileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timestamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params, "model_fig_drt":model_fig_drt}
    with open(f"{pklPath}\\{fileName}", "wb") as f:
        pickle.dump(checkpoint, f)

    # torch.save() -> .pt
    ptPath = Path(f"{root}\\model_registry\\pt\\")
    ptPath.mkdir(parents=True, exist_ok=True)
    ptFileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timestamp}.pt" 
    torch.save(network, f"{ptPath}\\{ptFileName}")

    # insert data into mongoDB
    Id = requests.get(f"http://127.0.0.1:8001/model/deployments/id/max").json()+1
    res = requests.post("http://127.0.0.1:8001/model/deployments", json={"modelId" : Id, "modelName" : fileName, \
                    "trainedDataset":data_drt, "deployStatus":"revoking", "containerID": "None", \
                        "containerPort" : "None"})

def writeIntoModelRegistry_hw1(network, model_experiments_record, model_params, model_fig_drt):

    data_drt = model_params.kwargs["dataDirectory"]

    # create dir  
    pklPath = Path(f"{root}\\model_registry\\pkl\\")
    pklPath.mkdir(parents=True, exist_ok=True)

    timestamp = model_params.kwargs["timestamp"]
    modelType = model_params.kwargs["modelFile"][:-3]
    validLoss = str(model_experiments_record["experiments_record"]["valid"]["mean_loss"])[:5]
    fileName = f"{data_drt}_{modelType}_{validLoss}_{timestamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params, "model_fig_drt":model_fig_drt}
    with open(f"{pklPath}\\{fileName}", "wb") as f:
        pickle.dump(checkpoint, f)

    # torch.save() -> .pt
    ptPath = Path(f"{root}\\model_registry\\pt\\")
    ptPath.mkdir(parents=True, exist_ok=True)
    ptFileName = f"{data_drt}_{modelType}_{validLoss}_{timestamp}.pt" 
    torch.save(network, f"{ptPath}\\{ptFileName}")

    # insert data into mongoDB
    Id = requests.get(f"http://127.0.0.1:8001/model/deployments/id/max").json()+1
    res = requests.post("http://127.0.0.1:8001/model/deployments", json={"modelId" : Id, "modelName" : fileName, \
                    "trainedDataset":data_drt, "deployStatus":"revoking", "containerID": "None", \
                        "containerPort" : "None"})

    # 
    studentId = model_params.kwargs["studentId"]
    ptFileName = f"{data_drt}_{modelType}_{validLoss}_{timestamp}.pt" 
    dataDirectory = model_params.kwargs["dataDirectory"]
    trainLoss = str(model_experiments_record["experiments_record"]["train"]["mean_loss"])[:5]
    validLoss = str(model_experiments_record["experiments_record"]["valid"]["mean_loss"])[:5]

    new_data = {"ptFileName":ptFileName, "dataDirectory":dataDirectory, "trainLoss":trainLoss, "validLoss":validLoss}
    # new_data = {studentId:[{"ptFileName":ptFileName, "dataDirectory":dataDirectory, "trainLoss":trainLoss, "validLoss":validLoss}]}
    
    hwPath = Path(f"{root}\\hw\\hw1\\{studentId}.json")

    if hwPath.is_file():
        with open(f"{hwPath}", "r") as file:
            existed_data = json.load(file)
        with open(f"{hwPath}", "w") as file:
            existed_data[studentId].append(new_data)
            json.dump(existed_data, file)
            file.close()
    else:
        with open(f"{hwPath}", "w") as file:
            existed_data = {studentId:[]}
            existed_data[studentId].append(new_data)
            json.dump(existed_data, file)
            file.close()


    





        


def writeIntoDockerApps(modelFile):

    """
    move the selected model file(pt) to \\ASLFN\\docker_apps\\
    """

    # define src, dst path
    srcPath = Path(f"{root}\\model_registry\\pt")
    # dstPath = Path(f"{root}\\model_deploying\\")
    dstPath = Path(f"{root}\\ASLFN\\docker_apps")

    removePreviousModelPtFile(dstPath)

    # create dir
    dstPath.mkdir(parents=True, exist_ok=True)

    source = f"{srcPath}\\{modelFile}"
    destination = f"{dstPath}\\{modelFile}"

    shutil.copyfile(source, destination)

def removePreviousModelPtFile(dstPath):

    """
    remove previous model file(pt)
    """

    p = Path(dstPath).glob('**/*')
    file = [x for x in p if str(x)[-3:] == ".pt"]
    if len(file) == 0 : return
    else: file[0].unlink(missing_ok=True)
