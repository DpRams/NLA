import time
import pickle
import torch
import sys
import numpy as np
import pandas as pd
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

    # write data to deployment.csv
    df = pd.read_csv(f"{root}\\model_deploying\\deployment.csv")
    Id = len(df)
    df.loc[-1] = [Id, fileName, data_drt, "revoking", "None", "None"]
    df.to_csv(f"{root}\\model_deploying\\deployment.csv", index=None)


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
