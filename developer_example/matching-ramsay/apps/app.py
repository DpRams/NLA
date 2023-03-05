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

###################################
# 以下皆為待填區域
module_id = "ramsay"
module_name = f"matching-{module_id}"

def matching(network):

    times_enlarge, times_shrink = 0,0

    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
        
        network.acceptable = True
#         print("Matching Successful")
        return network

    else:

        while times_enlarge + times_shrink < network.model_params["matchingTimes"]:
            
            output, loss = network.forward()
            network_pre = copy.deepcopy(network)
            loss_pre = loss
            
            # Backward and check the loss performance of the network with new learning rate
            network.backward(loss)
            output, loss = network.forward()

            # Confirm whether the loss value of the adjusted network is smaller than the current one
            if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):

                network.acceptable = True
                #print("Matching Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.model_params["learningRate"] *= 1.2
            else: #loss > loss_pre
                
                if network.model_params["learningRate"] <= network.model_params["matchingLearningRateLowerBound"]:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.model_params["learningRate"] *= 0.7
    

    network.acceptable = False
#     print("Matching Failed")
    return network_pre

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
	uvicorn.run("app:app", host="127.0.0.1", port=8005, reload=True)

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