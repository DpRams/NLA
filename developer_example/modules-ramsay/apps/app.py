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

# @app.get("/train")
# def root():
#     return "127.0.0.1"

# @app.post("/train/matching-ramsay")
# async def pipeline_service(request: Request):

#     json_POST = await request.json() # 補上 await 才能正確執行
#     old_network_weight, old_network_model_param, (x, y), nb_node, nb_node_pruned, acceptable = json_POST["network"], json_POST["model_params"], json_POST["data"], json_POST["nb_node"], json_POST["nb_node_pruned"], json_POST["acceptable"]
#     old_network = YourCSI_s2(dataDirectory = old_network_model_param["dataDirectory"], \
#                         dataDescribing = old_network_model_param["dataDescribing"], \
#                         dataShape = old_network_model_param["dataShape"], \
#                         modelFile = old_network_model_param["modelFile"], \
#                         inputDimension = old_network_model_param["inputDimension"], \
#                         hiddenNode = nb_node, \
#                         outputDimension = old_network_model_param["outputDimension"], \
#                         weightInitialization = old_network_model_param["weightInitialization"], \
#                         activationFunction = old_network_model_param["activationFunction"], \
#                         lossFunction = old_network_model_param["lossFunction"], \
#                         optimizer = old_network_model_param["optimizer"], \
#                         learningRate = old_network_model_param["learningRate"], \
#                         betas = old_network_model_param["betas"], \
#                         eps = old_network_model_param["eps"], \
#                         weightDecay = old_network_model_param["weightDecay"], \
#                         initializingRule = old_network_model_param["initializingRule"], \
#                         initializingNumber = old_network_model_param["initializingNumber"], \
#                         learningGoal = old_network_model_param["learningGoal"], \
#                         selectingRule = old_network_model_param["selectingRule"], \
#                         matchingRule = old_network_model_param["matchingRule"], \
#                         matchingTimes = old_network_model_param["matchingTimes"], \
#                         matchingLearningGoal = old_network_model_param["matchingLearningGoal"], \
#                         matchingLearningRateLowerBound = old_network_model_param["matchingLearningRateLowerBound"], \
#                         crammingRule = old_network_model_param["crammingRule"], \
#                         reorganizingRule = old_network_model_param["reorganizingRule"], \
#                         regularizingTimes = old_network_model_param["regularizingTimes"], \
#                         regularizingStrength = old_network_model_param["regularizingStrength"], \
#                         regularizingLearningGoal = old_network_model_param["regularizingLearningGoal"], \
#                         regularizingLearningRateLowerBound = old_network_model_param["regularizingLearningRateLowerBound"])

        
#     old_network_weight = odct_list2tensor(old_network_weight)

#     old_network.net.load_state_dict(old_network_weight, strict=False)

#     old_network.net.setData(x, y)
#     old_network.net.acceptable = acceptable

#     def EU_LG_UA(network):

#         times_enlarge, times_shrink = 0,0

#         network.acceptable = False

#         network_pre = copy.deepcopy(network)
#         output, loss = network.forward()

#         if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
            
#             network.acceptable = True
#     #         print("Matching Successful")
#             return network

#         else:

#             while times_enlarge + times_shrink < network.model_params["matchingTimes"]:
                
#                 output, loss = network.forward()
#                 network_pre = copy.deepcopy(network)
#                 loss_pre = loss
                
#                 # Backward and check the loss performance of the network with new learning rate
#                 network.backward(loss)
#                 output, loss = network.forward()

#                 # Confirm whether the loss value of the adjusted network is smaller than the current one
#                 if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):

#                     network.acceptable = True
#                     #print("Matching Successful")
#                     return network

#                 elif loss <= loss_pre:
                    
#                     # Multiply the learning rate by 1.2
#                     times_enlarge += 1
#                     network.model_params["learningRate"] *= 1.2
#                 else: #loss > loss_pre
                    
#                     if network.model_params["learningRate"] <= network.model_params["matchingLearningRateLowerBound"]:

#                         # If true, set the acceptance of the network as False and return initial_netwrok
#                         network.acceptable = False
#                         #print("Matching Failed")
#                         return network_pre
                    
#                     # On the contrary, restore w and adjust the learning rate
#                     else:

#                         # Restore the parameter of the network
#                         network = copy.deepcopy(network_pre)
#                         times_shrink += 1
#                         network.model_params["learningRate"] *= 0.7
        

#         network.acceptable = False
#     #     print("Matching Failed")
#         return network_pre

#     old_network.net = EU_LG_UA(old_network.net)

#     new_network_state_dict = odct_tensor2list(old_network.net)

#     return {"new_network":new_network_state_dict, \
#             "model_params" : old_network.net.model_params, \
#             "nb_node" : old_network.net.nb_node, \
#             "nb_node_pruned" : old_network.net.nb_node_pruned, \
#             "acceptable" : old_network.net.acceptable}

# @app.post("/train/cramming-ramsay")
# async def pipeline_service(request: Request):

#     json_POST = await request.json() # 補上 await 才能正確執行
#     old_network_weight, old_network_model_param, (x, y), nb_node, nb_node_pruned = json_POST["network"], json_POST["model_params"], json_POST["data"], json_POST["nb_node"], json_POST["nb_node_pruned"]
#     old_network = YourCSI_s2(dataDirectory = old_network_model_param["dataDirectory"], \
#                         dataDescribing = old_network_model_param["dataDescribing"], \
#                         dataShape = old_network_model_param["dataShape"], \
#                         modelFile = old_network_model_param["modelFile"], \
#                         inputDimension = old_network_model_param["inputDimension"], \
#                         hiddenNode = nb_node, \
#                         outputDimension = old_network_model_param["outputDimension"], \
#                         weightInitialization = old_network_model_param["weightInitialization"], \
#                         activationFunction = old_network_model_param["activationFunction"], \
#                         lossFunction = old_network_model_param["lossFunction"], \
#                         optimizer = old_network_model_param["optimizer"], \
#                         learningRate = old_network_model_param["learningRate"], \
#                         betas = old_network_model_param["betas"], \
#                         eps = old_network_model_param["eps"], \
#                         weightDecay = old_network_model_param["weightDecay"], \
#                         initializingRule = old_network_model_param["initializingRule"], \
#                         initializingNumber = old_network_model_param["initializingNumber"], \
#                         learningGoal = old_network_model_param["learningGoal"], \
#                         selectingRule = old_network_model_param["selectingRule"], \
#                         matchingRule = old_network_model_param["matchingRule"], \
#                         matchingTimes = old_network_model_param["matchingTimes"], \
#                         matchingLearningGoal = old_network_model_param["matchingLearningGoal"], \
#                         matchingLearningRateLowerBound = old_network_model_param["matchingLearningRateLowerBound"], \
#                         crammingRule = old_network_model_param["crammingRule"], \
#                         reorganizingRule = old_network_model_param["reorganizingRule"], \
#                         regularizingTimes = old_network_model_param["regularizingTimes"], \
#                         regularizingStrength = old_network_model_param["regularizingStrength"], \
#                         regularizingLearningGoal = old_network_model_param["regularizingLearningGoal"], \
#                         regularizingLearningRateLowerBound = old_network_model_param["regularizingLearningRateLowerBound"])
    
        
#     old_network_weight = odct_list2tensor(old_network_weight)

#     old_network.net.load_state_dict(old_network_weight, strict=False)

#     old_network.net.setData(x, y)

#     def ri_sro(network):

#         torch.random.manual_seed(0)

#         # Find unsatisfied data : k
#         network_old = copy.deepcopy(network)
#         output, loss = network.forward()

#         undesired_index = torch.nonzero(torch.abs(output - network.y) > network.model_params["learningGoal"] + 1e-3, as_tuple = False)

#         """
#         [ [0] [1] [3] ] shape = (3,1), 代表有3筆非0的data。 演算法基礎上應只有一筆，thus undesired_index.shape[0] == 1
#         """
#         if undesired_index.shape[0] == 1: # if the undesired_index is one and only one.
            
#             assert undesired_index[0][0] == len(network.y) - 1, "理應只有最後一筆需要cramming，此處出現異常"

#             # Unsatisfied situation
#             ## Find the index of the unsatisfied data
#             k_data_num = undesired_index[0][0]
#             undesired_data = torch.reshape(network.x[k_data_num,:], [1,-1])

#             ## Remove the data that does not meet the error term
#             left_data = network.x[:k_data_num,:]
#             right_data = network.x[k_data_num+1:,:]
#             remain_tensor = torch.cat([left_data, right_data], dim=0)
            
#             ## Use the random method to find out the gamma and zeta
#             while True:
                
#                 ## Find m-vector gamma: r
#                 ## Use the random method to generate the gamma that can meet the conditions
#                 gamma = torch.rand(size = [1, network.x.shape[1]]).to(network.device)
#                 subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
#                 matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

#                 if torch.all(matmul_value != 0):
#                     break
            
#             while True:

#                 ## Find the tiny value: zeta
#                 ## Use the random method to generate the zeta that can meet the conditions
#                 zeta = torch.rand(size=[1]).to(network.device)

#                 if torch.all(torch.mul(torch.add(zeta, matmul_value), torch.sub(zeta, matmul_value))<0):
#                     break
            
#             #k_l = undesired_index[0][0]  ### 如果直接用output[-1]，則不需要k_l

#             ## The weight of input layer to hidden layer I
#             w10 = gamma
#             w11 = gamma
#             w12 = gamma

#             W1_new = torch.cat([w10,w11,w12], dim=0)

#             ## The bias of input layer to hidden layer I
#             matual_value = torch.mm(gamma, torch.t(undesired_data))

#             b10 = torch.sub(zeta, matual_value)
#             b11 = -1*matual_value
#             b12 = torch.sub(-1*zeta, matual_value)

#             b1_new = torch.reshape(torch.cat([b10,b11,b12],0),[3])

#             ## The weight of hidden layer I to input layer
#             #gap = network.y[k_data_num, k_l] - output[k_data_num, k_l]  ### gap = network.y[-1] - output[-1]
#             gap = network.y[-1] - output[-1]

#             wo0_value = gap/zeta
#             wo1_value = (-2*gap)/zeta
#             wo2_value = gap/zeta

#             Wo_new = torch.reshape(torch.cat([wo0_value, wo1_value, wo2_value], dim=0), [1,-1])

#             ## Add new neuroes to the network
#             network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight.data, W1_new]))
#             network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias.data, b1_new]))
#             network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight.data, Wo_new],dim=1))
            
#             network.nb_node += 3

#             output, loss = network.forward()

#             ## Determine if cramming is succesful and print out the corresponding information
#             if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
#                 network.acceptable = True
#                 print("Cramming successful")
#                 network.message = "Cramming successful"
#                 return network
#             else:
#                 return None

#     old_network.net = ri_sro(old_network.net)

#     new_network_state_dict = odct_tensor2list(old_network.net)

#     return {"new_network":new_network_state_dict, \
#             "model_params" : old_network.net.model_params, \
#             "nb_node" : old_network.net.nb_node, \
#             "nb_node_pruned" : old_network.net.nb_node_pruned, \
#             "acceptable" : old_network.net.acceptable}

@app.post("/train/reorganizing-ramsay")
async def pipeline_service(request: Request):

    json_POST = await request.json() # 補上 await 才能正確執行
    old_network_weight, old_network_model_param, (x, y), nb_node, nb_node_pruned = json_POST["network"], json_POST["model_params"], json_POST["data"], json_POST["nb_node"], json_POST["nb_node_pruned"]
    old_network = YourCSI_s2(dataDirectory = old_network_model_param["dataDirectory"], \
                        dataDescribing = old_network_model_param["dataDescribing"], \
                        dataShape = old_network_model_param["dataShape"], \
                        modelFile = old_network_model_param["modelFile"], \
                        inputDimension = old_network_model_param["inputDimension"], \
                        hiddenNode = nb_node, \
                        outputDimension = old_network_model_param["outputDimension"], \
                        weightInitialization = old_network_model_param["weightInitialization"], \
                        activationFunction = old_network_model_param["activationFunction"], \
                        lossFunction = old_network_model_param["lossFunction"], \
                        optimizer = old_network_model_param["optimizer"], \
                        learningRate = old_network_model_param["learningRate"], \
                        betas = old_network_model_param["betas"], \
                        eps = old_network_model_param["eps"], \
                        weightDecay = old_network_model_param["weightDecay"], \
                        initializingRule = old_network_model_param["initializingRule"], \
                        initializingNumber = old_network_model_param["initializingNumber"], \
                        learningGoal = old_network_model_param["learningGoal"], \
                        selectingRule = old_network_model_param["selectingRule"], \
                        matchingRule = old_network_model_param["matchingRule"], \
                        matchingTimes = old_network_model_param["matchingTimes"], \
                        matchingLearningGoal = old_network_model_param["matchingLearningGoal"], \
                        matchingLearningRateLowerBound = old_network_model_param["matchingLearningRateLowerBound"], \
                        crammingRule = old_network_model_param["crammingRule"], \
                        reorganizingRule = old_network_model_param["reorganizingRule"], \
                        regularizingTimes = old_network_model_param["regularizingTimes"], \
                        regularizingStrength = old_network_model_param["regularizingStrength"], \
                        regularizingLearningGoal = old_network_model_param["regularizingLearningGoal"], \
                        regularizingLearningRateLowerBound = old_network_model_param["regularizingLearningRateLowerBound"])
    

    old_network_weight = odct_list2tensor(old_network_weight)

    old_network.net.load_state_dict(old_network_weight, strict=False)

    old_network.net.setData(x, y)

    def ri_sro_reorganizing(network):
        
        limit = 1

        ## If the number of hidden node has already been 1, do the regularizing then. 
        if network.linear1.bias.shape[0] <= limit:
            network = regularizing(network)
            # network = network.regularizing()
    #         print("Reorganizing finished")
            return(network)
        
        else:
            
            ## Set up the k = 1, and p = the number of hidden node
            k = 1
            p = network.linear1.weight.data.shape[0]

            while True:

                ## If k > p, end of Process
                if k > p or p <= limit:
    #                 print("Reorganizing finished")
                    return(network)

                ## Else, Process is ongoing
                else:

                    ## Using the regularizing module to adjust the network
                    network = regularizing(network)
                    # network = network.regularizing()

                    ## Store the network and w
                    network_pre = copy.deepcopy(network)

                    ## Set up the acceptable of the network as false
                    network.acceptable = False
                    
                
                    ## Ignore the K hidden node
                    network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight[:k-1],network.linear1.weight[k:]],0))
                    network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias[:k-1],network.linear1.bias[k:]]))
                    network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight[:,:k-1],network.linear2.weight[:,k:]],1))

                    
                    ## Using the matching module to adjust the network
                    network = matching_for_reorganizing(network)

                    ## If the resulting network is acceptable, this means that the k hidden node can be removed
                    if network.acceptable:

                        network.nb_node -= 1
                        network.nb_node_pruned += 1
                        p-=1

                    ## Else, it means that the k hidden node cannot be removed
                    else:

                        ## Restore the network and w
                        network = copy.deepcopy(network_pre)
                        k+=1

    def matching_for_reorganizing(network):

        times_enlarge, times_shrink = 0, 0

        # Set up the learning rate of the network
        # network.model_params["learningRate"] = 1e-3
        network.acceptable = False

        network_pre = copy.deepcopy(network)
        output, loss = network.forward()

        if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
            
            network.acceptable = True
    #         print("Matching(Re) Successful")
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
                    #print("Matching(Re) Successful")
                    return network

                elif loss <= loss_pre:
                    
                    # Multiply the learning rate by 1.2
                    times_enlarge += 1
                    network.model_params["learningRate"] *= 1.2
                else: #loss > loss_pre
                    
                    if network.model_params["learningRate"] <= network.model_params["matchingLearningRateLowerBound"]:

                        # If true, set the acceptance of the network as False and return initial_netwrok
                        network.acceptable = False
                        #print("Matching(Re) Failed")
                        return network_pre
                    
                    # On the contrary, restore w and adjust the learning rate
                    else:

                        # Restore the parameter of the network
                        network = copy.deepcopy(network_pre)
                        times_shrink += 1
                        network.model_params["learningRate"] *= 0.7
        

        network.acceptable = False
    #     print("Matching(Re) Failed")
        return network_pre       

    def regularizing(network):
        
        ## Record the number of executions
        times_enlarge, times_shrink = 0, 0

        # ## Set up the learning rate of the network
        # network.model_params["learningRate"] = 1e-3

        ## Set epoch to 100
        for i in range(network.model_params["matchingTimes"]):

            ## Store the parameter of the network
            network_pre = copy.deepcopy(network)
            output, loss = network.forward(network.model_params["regularizingStrength"])
            loss_pre = loss

            ## Backward operation to optain w'
            network.backward(loss)
            output, loss = network.forward(network.model_params["regularizingStrength"])

            ## Confirm whether the adjusted loss value is smaller than the current one
            if loss <= loss_pre:

                ## Identify that all forecast value has met the error term
                if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
                    
                    ## If true, multiply the learning rate by 1.2
                    network.model_params["learningRate"] *= 1.2
                    times_enlarge += 1

                else:

                    ## Else, restore w and end the process
                    network = copy.deepcopy(network_pre)
                    return network

            # If the adjusted loss value is not smaller than the current one
            else:

                ## If the learning rate is greater than the threshold for learning rate
                if network.model_params["learningRate"] > network.model_params["regularizingLearningRateLowerBound"]:
                    
                    ## Restore the w and multiply the learning rate by 0.7
                    network = copy.deepcopy(network_pre)
                    network.model_params["learningRate"] *= 0.7
                    times_shrink += 1

                ## If the learning rate is smaller than the threshold for learning rate
                else:
                    
                    ## Restore the w
                    network = copy.deepcopy(network_pre)
    #                 print("Regularizing finished")
                    return network
                    
        #print("Regularizing finished")
        return network

    old_network.net = ri_sro_reorganizing(old_network.net)

    new_network_state_dict = odct_tensor2list(old_network.net)

    return {"new_network":new_network_state_dict, \
            "model_params" : old_network.net.model_params, \
            "nb_node" : old_network.net.nb_node, \
            "nb_node_pruned" : old_network.net.nb_node_pruned, \
            "acceptable" : old_network.net.acceptable}

if __name__ == '__main__':
	uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True) # 若有 rewrite file 可能不行 reload=True，不然會一直重開 by 李忠大師

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