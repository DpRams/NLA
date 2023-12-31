from fastapi import FastAPI, Request, Form
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from network.net import Network
from modelParameter import ModelParameter

import uvicorn
import numpy as np
import torch 
import pickle
import os
import pandas as pd
import sys
import time

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

app = FastAPI()


def reading_pkl(modelFile):
    
    modelPath = f"{modelFile}"
    with open(modelPath, 'rb') as f:
        print(f)
        checkpoints = pickle.load(f)

    return checkpoints 


def inferencing(network, x_test, y_test, validating=False):   

    network.eval()
    
    network.setData(x_test, y_test, only_cpu=True)
    output, loss = network.forward()

    if not validating:
        return np.round(loss.detach().cpu().numpy(), 3)
        
    else:
        learningGoal = network.model_params["learningGoal"]  
        diff = output - network.y 
        acc = (diff <= learningGoal).to(torch.float32).mean().cpu().numpy()
        return np.round(acc, 3)


global network

p = Path(".").glob('**/*')
modelPklFile = str([x for x in p if x.is_file() and str(x).endswith(".pt")][0])
# network = torch.load(modelPklFile)
network = torch.load(modelPklFile, map_location=torch.device("cpu"))
# network = reading_pkl(modelPklFile)["model_experiments_record"]["network"]

@app.get("/")
def pipeline_service():

    return "GET 127.0.0.1:8002/"


@app.get("/predict")
def pipeline_service(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelPklFile: str = Form(default=None, max_length=50)):

    return "GET 127.0.0.1:8002/predict"
    
# 換 pkl 測試，記得要重啟服務才會 load 新的 pkl
@app.post("/predict")
async def pipeline_service(request: Request):
    
    start = time.time()

    global network

    json_POST = await request.json() # 補上 await 才能正確執行
    x_test, y_test = json_POST["dataDirectory"]["x_test"], json_POST["dataDirectory"]["y_test"]
    rmseError = inferencing(network, x_test, y_test)


    return {"Message" : "POST 127.0.0.1:8002/predict", \
            # "dataDirectory" : json_POST["dataDirectory"], \
            "rmseError" : str(rmseError), \
            "time" : str(time.time()-start)
            }



if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
    
    
