from fastapi import FastAPI, Request
from net import RIRO, YourCSI_s2
from modelParameter import ModelParameter
import uvicorn
import torch
import numpy as np 
import copy


app = FastAPI()

"""
作為 module API，等待被呼叫訓練使用
"""

def odct_tensor2list(network):

    old_network = {}
    for dct in network.state_dict().items():
        key, value = dct
        old_network[key] = value.tolist()
    return old_network

def odct_list2tensor(network_weight):

    network_weight = {}
    for dct in network_weight.items():
        key, value = dct
        network_weight[key] = torch.FloatTensor(value)
    return network_weight

###################################
# 以下皆為待填區域

# Step 1. 用自己的學號複寫 module_id 的變數值
module_id = "default"
module_name = f"matching-{module_id}"

def matching(network):
    
    # Step 2. 此為待填區域

    return network

# Step 3. 將資料夾名稱更改為 module_name 的變數值，如 : matching-jason
# Step 4. 上傳資料夾
# 以上皆為待填區域
###################################



@app.post(f"/train/{module_name}")
async def pipeline_service(request: Request):

    network_related_info = await request.json()

    network_weight = odct_list2tensor(network_related_info["network"])
    network_model_params = network_related_info["model_params"]
    (x_train, y_train) = network_related_info["data"]
    nb_node = network_related_info["nb_node"]
    nb_node_pruned = network_related_info["nb_node_pruned"]
    acceptable = network_related_info["acceptable"]
    
    CSI = YourCSI_s2(dataDirectory = network_model_params["dataDirectory"], \
                        dataDescribing = network_model_params["dataDescribing"], \
                        dataShape = network_model_params["dataShape"], \
                        modelFile = network_model_params["modelFile"], \
                        inputDimension = network_model_params["inputDimension"], \
                        hiddenNode = nb_node, \
                        outputDimension = network_model_params["outputDimension"], \
                        weightInitialization = network_model_params["weightInitialization"], \
                        activationFunction = network_model_params["activationFunction"], \
                        lossFunction = network_model_params["lossFunction"], \
                        optimizer = network_model_params["optimizer"], \
                        learningRate = network_model_params["learningRate"], \
                        betas = network_model_params["betas"], \
                        eps = network_model_params["eps"], \
                        weightDecay = network_model_params["weightDecay"], \
                        initializingRule = network_model_params["initializingRule"], \
                        initializingNumber = network_model_params["initializingNumber"], \
                        learningGoal = network_model_params["learningGoal"], \
                        selectingRule = network_model_params["selectingRule"], \
                        matchingRule = network_model_params["matchingRule"], \
                        matchingTimes = network_model_params["matchingTimes"], \
                        matchingLearningGoal = network_model_params["matchingLearningGoal"], \
                        matchingLearningRateLowerBound = network_model_params["matchingLearningRateLowerBound"], \
                        crammingRule = network_model_params["crammingRule"], \
                        reorganizingRule = network_model_params["reorganizingRule"], \
                        regularizingTimes = network_model_params["regularizingTimes"], \
                        regularizingStrength = network_model_params["regularizingStrength"], \
                        regularizingLearningGoal = network_model_params["regularizingLearningGoal"], \
                        regularizingLearningRateLowerBound = network_model_params["regularizingLearningRateLowerBound"])

    # restore the net
    CSI.net.load_state_dict(network_weight, strict=False)
    CSI.net.setData(x_train, y_train)
    CSI.net.acceptable = acceptable
    CSI.net = matching(CSI.net)
    network_weight = odct_tensor2list(CSI.net)

    return {"new_network":network_weight, \
            "model_params" : CSI.net.model_params, \
            "nb_node" : CSI.net.nb_node, \
            "nb_node_pruned" : CSI.net.nb_node_pruned, \
            "acceptable" : CSI.net.acceptable}

if __name__ == '__main__':
	uvicorn.run("app:app", host="0.0.0.0", port=8005, reload=True)

