import time
import pickle
import torch
import sys
import numpy as np
from pathlib import Path
import shutil
from network.net import Network

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

def writeIntoModelRegistry(model_experiments_record, model_params, model_fig_drt):

    data_drt = model_params.kwargs["dataDirectory"]
    
    if "initializingLearningGoal" in model_params.kwargs.keys():
        lr_goal = model_params.kwargs["initializingLearningGoal"]
    else:
        lr_goal = model_params.kwargs["learningGoal"]

    # create dir  
    pklPath = Path(f"{root}\\model_registry\\pkl\\")
    pklPath.mkdir(parents=True, exist_ok=True)

    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.kwargs["modelFile"][:-3]
    validAcc = str(model_experiments_record["experiments_record"]["valid"]["mean_acc"])[:5]
    fileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params, "model_fig_drt":model_fig_drt}
    with open(f"{pklPath}\\{fileName}", "wb") as f:
        pickle.dump(checkpoint, f)

    # torch.save() -> .pt
    ptPath = Path(f"{root}\\model_registry\\pt\\")
    ptPath.mkdir(parents=True, exist_ok=True)
    ptFileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}.pt" 
    torch.save(model_experiments_record["network"],  f"{ptPath}\\{ptFileName}")


def writeIntoModelDeploying(modelFile):

    

    # define src, dst path
    srcPath = Path(f"{root}\\model_registry\\pt")
    # dstPath = Path(f"{root}\\model_deploying\\")
    dstPath = Path(f"{root}\\ASLFN\\docker_apps")

    removePreviousModelPtFile(dstPath)

    # # create dir
    # dstPath.mkdir(parents=True, exist_ok=True)

    # source = f"{srcPath}\\{modelFile}"
    # destination = f"{dstPath}\\{modelFile}"

    # shutil.copyfile(source, destination)

def removePreviousModelPtFile(dstPath):

    p = Path(dstPath).glob('**/*')
    files = [x for x in p if str(x)[-3:] == ".pt"]
    print(files)



    pass
