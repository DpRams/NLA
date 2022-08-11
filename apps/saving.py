import time
import pickle
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


def writeIntoModelRegistry(model_experiments_record, model_params, model_fig_drt):

    data_drt = model_params.dataDirectory
    lr_goal = model_params.learningGoal

    drtPath = f"{root}\\model_registry\\"
    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.modelFile[:-3]
    validAcc = model_experiments_record["lr_goals"][lr_goal]["experiments_record"]["valid"]["mean_acc"]
    fileName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params, "model_fig_drt":model_fig_drt}
    with open(drtPath + fileName, "wb") as f:
        pickle.dump(checkpoint, f)