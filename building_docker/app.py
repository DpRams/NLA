from fastapi import FastAPI, Request, Form
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from pathlib import Path


import uvicorn
import numpy as np
import torch 
import pickle
import os
import pandas as pd
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

app = FastAPI()

@app.get("/predict")
def pipeline_service(request: Request, \
                     dataDirectory: str = Form(default=None, max_length=50), \
                     modelPklFile: str = Form(default=None, max_length=50)):

    return "GET 127.0.0.1:8001/predict"

@app.post("/predict")
def pipeline_service(request: Request):

    # x_test, y_test = reading_dataset_Testing(dataDirectory)
    # checkpoints = reading_pkl(modelPklFile)
    # network = checkpoints["model_experiments_record"]["network"]
    # rmseError = inferencing(network, x_test, y_test)

    # json_d = request.json() # 這段有問題


    rmseError = 0.01

    return {"Message" : "POST 127.0.0.1:8001/predict", "rmseError" : rmseError}

#    model_experiments_record, model_params, model_fig_drt, rmseError = __model_evaluating(dataDirectory, modelPklFile)

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
    
    modelPath = f"{root}/model_registry/{modelFile}"
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



if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True) # 若有 rewrite file 可能不行 reload=True，不然會一直重開 by 李忠大師
