from pyexpat import model
import time
import pickle
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# commit message沒寫好，再push一次

def writeIntoModelRegistry(model_experiments_record, model_params):

    drtPath = f"{root}\\model_registry\\"
    timeStamp = time.strftime("%Y%d%m_%H%M%S", time.localtime())
    modelType = model_params.modelFile[:-3]
    validAcc = model_experiments_record["lr_goals"][model_params.learningGoal]["experiments_record"]["valid"]["mean_acc"]
    fileName = f"{model_params.dataDirectory}_{modelType}_{model_params.learningGoal}_{validAcc}_{timeStamp}.pkl" 
    checkpoint = {"model_experiments_record":model_experiments_record, "model_params":model_params}
    with open(drtPath + fileName, "wb") as f:
        pickle.dump(checkpoint, f)