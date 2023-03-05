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
module_name = f"cramming-{module_id}"

def cramming(network):

    torch.random.manual_seed(0)

    # Find unsatisfied data : k
    network_old = copy.deepcopy(network)
    output, loss = network.forward()

    undesired_index = torch.nonzero(torch.abs(output - network.y) > network.model_params["learningGoal"] + 1e-3, as_tuple = False)

    """
    [ [0] [1] [3] ] shape = (3,1), 代表有3筆非0的data。 演算法基礎上應只有一筆，thus undesired_index.shape[0] == 1
    """
    if undesired_index.shape[0] == 1: # if the undesired_index is one and only one.
        
        assert undesired_index[0][0] == len(network.y) - 1, "理應只有最後一筆需要cramming，此處出現異常"

        # Unsatisfied situation
        ## Find the index of the unsatisfied data
        k_data_num = undesired_index[0][0]
        undesired_data = torch.reshape(network.x[k_data_num,:], [1,-1])

        ## Remove the data that does not meet the error term
        left_data = network.x[:k_data_num,:]
        right_data = network.x[k_data_num+1:,:]
        remain_tensor = torch.cat([left_data, right_data], dim=0)
        
        ## Use the random method to find out the gamma and zeta
        while True:
            
            ## Find m-vector gamma: r
            ## Use the random method to generate the gamma that can meet the conditions
            gamma = torch.rand(size = [1, network.x.shape[1]]).to(network.device)
            subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
            matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

            if torch.all(matmul_value != 0):
                break
        
        while True:

            ## Find the tiny value: zeta
            ## Use the random method to generate the zeta that can meet the conditions
            zeta = torch.rand(size=[1]).to(network.device)

            if torch.all(torch.mul(torch.add(zeta, matmul_value), torch.sub(zeta, matmul_value))<0):
                break
        
        #k_l = undesired_index[0][0]  ### 如果直接用output[-1]，則不需要k_l

        ## The weight of input layer to hidden layer I
        w10 = gamma
        w11 = gamma
        w12 = gamma

        W1_new = torch.cat([w10,w11,w12], dim=0)

        ## The bias of input layer to hidden layer I
        matual_value = torch.mm(gamma, torch.t(undesired_data))

        b10 = torch.sub(zeta, matual_value)
        b11 = -1*matual_value
        b12 = torch.sub(-1*zeta, matual_value)

        b1_new = torch.reshape(torch.cat([b10,b11,b12],0),[3])

        ## The weight of hidden layer I to input layer
        #gap = network.y[k_data_num, k_l] - output[k_data_num, k_l]  ### gap = network.y[-1] - output[-1]
        gap = network.y[-1] - output[-1]

        wo0_value = gap/zeta
        wo1_value = (-2*gap)/zeta
        wo2_value = gap/zeta

        Wo_new = torch.reshape(torch.cat([wo0_value, wo1_value, wo2_value], dim=0), [1,-1])

        ## Add new neuroes to the network
        network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight.data, W1_new]))
        network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias.data, b1_new]))
        network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight.data, Wo_new],dim=1))
        
        network.nb_node += 3

        output, loss = network.forward()

        ## Determine if cramming is succesful and print out the corresponding information
        if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
            network.acceptable = True
            print("Cramming successful")
            network.message = "Cramming successful"
            return network
        else:
            return None

# 以上皆為待填區域
###################################

@app.post("/train/cramming-ramsay")
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
    CSI.net = cramming(CSI.net)
    
    network_weight = odct_tensor2list(CSI.net)

    return {"new_network":network_weight, \
            "model_params" : CSI.net.model_params, \
            "nb_node" : CSI.net.nb_node, \
            "nb_node_pruned" : CSI.net.nb_node_pruned, \
            "acceptable" : CSI.net.acceptable}


if __name__ == '__main__':
	uvicorn.run("app:app", host="0.0.0.0", port=8005, reload=True) # 若有 rewrite file 可能不行 reload=True，不然會一直重開 by 李忠大師

def odct_tensor2list(network):

    old_network = {}
    for dct in network.state_dict().items():
        key, value = dct
        old_network[key] = value.tolist()
    return old_network

def odct_list2tensor(old_network_weight):

    network_weight = {}
    for dct in old_network_weight.items():
        key, value = dct
        network_weight[key] = torch.FloatTensor(value)
    return network_weight

