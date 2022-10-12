from fastapi import FastAPI, Request, Form
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from pathlib import Path

from net import Network
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

def reading_dataset_Testing(dataDirecotry):
    
    filelist = os.listdir(f"{root}/upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    print(f"filePath_X = {filePath_X}\nfilePath_Y = {filePath_Y}")
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    return (X_transformed, Y_transformed)

def reading_pkl(modelFile):
    
    modelPath = f"{modelFile}"
    with open(modelPath, 'rb') as f:
        checkpoints = pickle.load(f)

    return checkpoints 



def inferencing(network, x_test, y_test, validating=False):   

    network.eval()
    
    network.setData(x_test, y_test)
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
modelPklFile = str([x for x in p if x.is_file() and str(x).endswith(".pkl")][0])
network = reading_pkl(modelPklFile)["model_experiments_record"]["network"]

@app.get("/predict")
def pipeline_service(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelPklFile: str = Form(default=None, max_length=50)):

    return "GET 127.0.0.1:8001/predict"

@app.post("/predict")
async def pipeline_service(request: Request):
    
    start = time.time()

    global network
    dataDirectory = "solar"
    x_test, y_test = reading_dataset_Testing(dataDirectory)
    rmseError = inferencing(network, x_test, y_test)

    json_POST = await request.json() # 補上 await 才能正確執行

    # rmseError = 0.01

    print(network)
    print(rmseError)


    return {"Message" : "POST 127.0.0.1:8001/predict", \
            "dataDirectory" : json_POST["dataDirectory"], \
            "modelPklFile" : json_POST["modelPklFile"], \
            "rmseError" : str(rmseError), \
            "time" : str(time.time()-start)
            }

#    model_experiments_record, model_params, model_fig_drt, rmseError = __model_evaluating(dataDirectory, modelPklFile)




if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
    
    
