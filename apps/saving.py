import time
import pickle
import sys
import numpy as np
from pathlib import Path

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
    drtPath = Path(f"{root}\\model_registry\\")
    drtPath.mkdir(parents=True, exist_ok=True)

    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.kwargs["modelFile"][:-3]
    validAcc = str(model_experiments_record["experiments_record"]["valid"]["mean_acc"])[:5]
    fileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params, "model_fig_drt":model_fig_drt}
    with open(f"{drtPath}\\{fileName}", "wb") as f:
        pickle.dump(checkpoint, f)