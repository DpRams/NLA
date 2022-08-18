# def main():
#     print("Call riro.py main function")


import torch 
import copy
import pickle
import os 
import sys

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
# sys.path.append(str(parent))
sys.path.append(str(root))

from apps import evaluating, saving

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(torch.nn.Module):

    def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):

        super().__init__()
        # print(f"參數灌入:{kwargs}")
        # Initialize
        self.linear1 = torch.nn.Linear(x_train_scaled.shape[1], nb_neuro).to(device)
        self.linear2 = torch.nn.Linear(nb_neuro, 1).to(device)
    
        # Stop criteria - threshold
        self.threshold_for_error = eval(kwargs["learning_goal"]) 
        self.threshold_for_lr = eval(kwargs["learning_rate_lower_bound"]) 
        self.tuning_times = eval(kwargs["tuning_times"]) 
        self.regularizing_strength = eval(kwargs["regularizing_strength"])
        
        # Set default now, not open for customization.
        not_used_currently = (kwargs["regularizing_strength"], kwargs["optimizer"])

        # Input data
        self.x = torch.FloatTensor(x_train_scaled).to(device)
        self.y = torch.FloatTensor(y_train_scaled).to(device)

        # Learning rate
        self.learning_rate = eval(kwargs["learning_rate"])

        # Whether the network is acceptable, default as False
        self.acceptable = False

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = nb_neuro
        
        self.undesired_index = None
        self.message = ""
        
    # Reset the x and y data
    def setData(self, x_train_scaled, y_train_scaled):
        
        self.x = torch.FloatTensor(x_train_scaled).to(device)
        self.y = torch.FloatTensor(y_train_scaled).to(device)
    # Add the new data to the x and y data
    def addData(self, new_x_train, new_y_train):
        
        self.x = torch.cat([self.x, new_x_train.reshape(1,-1)], 0)#.cuda()
        self.y = torch.cat([self.y, new_y_train.reshape(-1,1)], 0)#.cuda()

    # Forward operaion
    def forward(self, reg_strength=0):
        
        h_relu = self.linear1(self.x).clamp(min=0)
        output = self.linear2(h_relu)

        param_val = torch.sum(torch.pow(self.linear2.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear2.weight.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.weight.data, 2))
        reg_term = reg_strength / (
            self.linear1.weight.data.shape[1] + 1 \
            + self.linear1.weight.data.shape[1] * (self.linear1.weight.data.shape[0] + 1)
        ) * param_val

        loss = torch.sqrt(torch.nn.functional.mse_loss(output, self.y)) + reg_term

        return (output, loss)
    
    # Backward operation
    def backward_Adam(self, loss):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def initializing(network, initial_x, initial_y):
    
    # Find each minimum output value y
    min_y = min(initial_y)

    # Subtract min_y from each y
    res_y = initial_y - min_y
    
    # Use Linear regression to find the initial W1, b1, W2, b2
    reg = LinearRegression().fit(initial_x, res_y)

    # Set up the initial parameter of the network(through torch.nn.Parameter, we can turn non-trainable tensor into trainable tensor)
    network.linear1.weight = torch.nn.Parameter(torch.FloatTensor(reg.coef_).to(device))
    network.linear1.bias = torch.nn.Parameter(torch.FloatTensor(reg.intercept_).to(device))
    network.linear2.weight = torch.nn.Parameter(torch.FloatTensor([[1]]).to(device))
    print(f'torch.FloatTensor(min_y).to(device) = {torch.FloatTensor(min_y).to(device)}')
    network.linear2.bias = torch.nn.Parameter(torch.FloatTensor(min_y).to(device))
    
    network.acceptable = True

    return network

def selecting(network, x_train_scaled, y_train_scaled):
    
    residual = []
    temp_network = copy.deepcopy(network)

    # Put each data into network to calculate the loss value
    for i in range(x_train_scaled.shape[0]):
        temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
        residual.append((temp_network.forward()[1].item(),i))
    
    
    # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
    sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
    return sorted_index

def matching(network):

    global tuning_times
    times_enlarge, times_shrink = 0,0

    # Set up the learning rate of the network
    # network.learning_rate = 1e-3
    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
        
        network.acceptable = True
#         print("Matching Successful")
        return network

    else:

        while times_enlarge + times_shrink < network.tuning_times:
            
            output, loss = network.forward()
            network_pre = copy.deepcopy(network)
            loss_pre = loss
            
            # Backward and check the loss performance of the network with new learning rate
            network.backward_Adam(loss)
            output, loss = network.forward()

            # Confirm whether the loss value of the adjusted network is smaller than the current one
            if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.threshold_for_error):

                network.acceptable = True
                #print("Matching Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.learning_rate *= 1.2
            else: #loss > loss_pre
                
                if network.learning_rate <= network.threshold_for_lr:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.learning_rate *= 0.7
      

    network.acceptable = False
#     print("Matching Failed")
    return network_pre       

def matching_for_reorganizing(network):

    times_enlarge, times_shrink = 0, 0

    # Set up the learning rate of the network
    # network.learning_rate = 1e-3
    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
        
        network.acceptable = True
#         print("Matching(Re) Successful")
        return network

    else:

        while times_enlarge + times_shrink < network.tuning_times:
            
            output, loss = network.forward()
            network_pre = copy.deepcopy(network)
            loss_pre = loss
            
            # Backward and check the loss performance of the network with new learning rate
            network.backward_Adam(loss)
            output, loss = network.forward()

            # Confirm whether the loss value of the adjusted network is smaller than the current one
            if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.threshold_for_error):

                network.acceptable = True
                #print("Matching(Re) Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.learning_rate *= 1.2
            else: #loss > loss_pre
                
                if network.learning_rate <= network.threshold_for_lr:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching(Re) Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.learning_rate *= 0.7
      

    network.acceptable = False
#     print("Matching(Re) Failed")
    return network_pre         

def cramming(network): # 把undesired_index print出來，看一下k_data_num(un..[0][0]), k_l(un..[0][1])
    """
    其實undesired_index的作用應該可以單純用network.y[-1]來取代，因為每次都在處理最新的一筆資料
    """
    torch.random.manual_seed(0)

    # Find unsatisfied data : k
    network_old = copy.deepcopy(network)
    output, loss = network.forward()

    undesired_index = torch.nonzero(torch.abs(output - network.y) > network.threshold_for_error + 1e-3, as_tuple = False)

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
            gamma = torch.rand(size = [1, network.x.shape[1]]).to(device)
            subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
            matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

            if torch.all(matmul_value != 0):
                break
        
        while True:

            ## Find the tiny value: zeta
            ## Use the random method to generate the zeta that can meet the conditions
            zeta = torch.rand(size=[1]).to(device)

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
        if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
            network.acceptable = True
            print("Cramming successful")
            network.message = "Cramming successful"
            return network
        else:
#             print("Cramming failed，視該筆資料為離群值，無法學習")
            network = copy.deepcopy(network_old)
            network.message = "Cramming failed，視該筆資料為離群值，無法學習"
            network.undesired_index = undesired_index[:,0].cpu()
            return network
    elif undesired_index.shape[0] == 0:
        network.message = "資料皆滿足learning_goal"
        return network
    else:
#         print("有多筆 undesired_index，無法cramming")
        print(f'undesired_index = {undesired_index}')
        network.message = "有多筆 undesired_index，無法cramming"
        network.undesired_index = undesired_index[:,0].cpu()
        return network


def regularizing(network):
    
    global tuning_times
    ## Record the number of executions
    times_enlarge, times_shrink = 0, 0

    # ## Set up the learning rate of the network
    # network.learning_rate = 1e-3

    ## Set epoch to 100
    for i in range(network.tuning_times):

        ## Store the parameter of the network
        network_pre = copy.deepcopy(network)
        output, loss = network.forward(network.regularizing_strength)
        loss_pre = loss

        ## Backward operation to optain w'
        network.backward_Adam(loss)
        output, loss = network.forward(network.regularizing_strength)

        ## Confirm whether the adjusted loss value is smaller than the current one
        if loss <= loss_pre:

            ## Identify that all forecast value has met the error term
            if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
                
                ## If true, multiply the learning rate by 1.2
                network.learning_rate *= 1.2
                times_enlarge += 1

            else:

                ## Else, restore w and end the process
                network = copy.deepcopy(network_pre)
                return(network)

        # If the adjusted loss value is not smaller than the current one
        else:

            ## If the learning rate is greater than the threshold for learning rate
            if network.learning_rate > network.threshold_for_lr:
                
                ## Restore the w and multiply the learning rate by 0.7
                network = copy.deepcopy(network_pre)
                network.learning_rate *= 0.7
                times_shrink += 1

            ## If the learning rate is smaller than the threshold for learning rate
            else:
                
                ## Restore the w
                network = copy.deepcopy(network_pre)
#                 print("Regularizing finished")
                return(network)
                
    #print("Regularizing finished")
    return(network)

            

def reorganizing(network):
    

    limit = 1

    ## If the number of hidden node has already been 1, do the regularizing then. 
    if network.linear1.bias.shape[0] <= limit:
        network = regularizing(network)
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



def inferencing(network, x_test, y_test):   

    network.eval()
    
    network.setData(x_test, y_test)
    output, loss = network.forward()
        
    diff = (output - network.y)
    acc = (diff <= network.threshold_for_error).to(torch.float32).mean().cpu().numpy()
    return acc


# Training Accuracy, Training Loss, nb_node, nb_node_pruned 折線圖
# Route 圓餅圖
def plot_loss(checkpoint):
    
    plt.plot([i.cpu().detach() for i in checkpoint['train_loss_history']], label = 'train') 
    plt.plot(checkpoint['val_loss_history'], label = "val") 
    plt.title("Training / Val loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def reading_dataset_Training(dataDirecotry):
    
    filelist = os.listdir(f"./upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    print(f"filePath_X = {filePath_X}\nfilePath_Y = {filePath_Y}")
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    # train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, Y_transformed, test_size=0.2, random_state=42)

    # Split data into intializing use and training use.
    initial_x, x_train_scaled = torch.FloatTensor(x_train[:x_train.shape[1]+1]), torch.FloatTensor(x_train[x_train.shape[1]+1:])
    initial_y, y_train_scaled = torch.FloatTensor(y_train[:x_train.shape[1]+1]), torch.FloatTensor(y_train[x_train.shape[1]+1:])

    # print(f'initial_x.shape : {initial_x.shape}')
    # print(f'initial_y.shape : {initial_y.shape}\n')
    # print(f'x_train_scaled.shape : {x_train_scaled.shape}')
    # print(f'y_train_scaled.shape : {y_train_scaled.shape}\n')
    
    return (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test)

def __is_lr_goal_big_enough(network):
    
    if network.message == "有多筆 undesired_index，無法cramming" or network.message == "Cramming failed，視該筆資料為離群值，無法學習": 
        return True
    else:
        return False
    

# 存放model, experiments_record
def main(model_params):

    lr_goals = [model_params.learningGoal]
    # model_experiments_record = {"lr_goals" : {key : None for key in lr_goals}}
    model_experiments_record = {"network" : None, "experiments_record" : None}

    for lr_goal in sorted(lr_goals, reverse=True):
        
        # Reading dataset
        (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test) = reading_dataset_Training(model_params.dataDirectory)
        
        # Defining model
        network = Network(1, initial_x, initial_y, \
                            loss_function=model_params.lossFunction, \
                            learning_goal=lr_goal, \
                            learning_rate=model_params.learningRate, \
                            learning_rate_lower_bound=model_params.learningRateLowerBound, \
                            optimizer=model_params.optimizer,  \
                            tuning_times=model_params.tuningTimes,  \
                            regularizing_strength=model_params.regularizingStrength)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = network.to(device)

        # Initializing model
        network = initializing(network, initial_x, initial_y)

        # Checking if lr_goal is too small
        # if __is__lr_goal_big_enough():pass
        # else:return "lr_goal is too small to training!"

        # Record experiments data
        experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                              "valid" : {"mean_acc" : 0}, \
                              "nb_node" : [], "nb_node_pruned" : [],\
                              "Route" : {"Blue": 0, "Red":0, "Green":0, "Purple":0}}

        experiments_record["nb_node"].append(network.linear1.weight.data.shape[0])
        
        # The initializing-use data should be add into the training data too.
        current_x, current_y = initial_x, initial_y 

        for i in range(1, x_train_scaled.shape[0]):
        
            print('-----------------------------------------------------------')
            print(f"訓練第幾筆資料 : {i + x_train_scaled.shape[1] + 1}")

            sorted_index = selecting(network, x_train_scaled[i-1:], y_train_scaled[i-1:])
            current_x = np.append(current_x, x_train_scaled[sorted_index[0]]).reshape(-1, x_train_scaled.shape[1])
            current_y = np.append(current_y, y_train_scaled[sorted_index[0]].reshape(-1, 1))
            current_y = np.expand_dims(current_y, 1) #turn shape [n] into [n,1] 

            print(f'current_x = {current_x.shape}')
            print(f'current_y = {current_y.shape}')

            network.setData(current_x, current_y)
            network_pre = copy.deepcopy(network)

            output, loss = network.forward()

            if torch.all(torch.abs(output - network.y) <= network.threshold_for_error):

                network.acceptable = True
                network = reorganizing(network)
                experiments_record["Route"]["Blue"] += 1
            else:

                network.acceptable = False
                network = matching(network)

                if network.acceptable:

                    network = reorganizing(network)
                    experiments_record["Route"]["Green"] += 1

                else:

                    network = copy.deepcopy(network_pre)
                    network = cramming(network)
                    print("Cramming End")
                    
                    if  __is_lr_goal_big_enough(network):
                        return "lr_goal is too small to training!", "lr_goal is too small to training!", "lr_goal is too small to training!"
                    
                    network = reorganizing(network)

                    experiments_record["Route"]["Red"] += 1

            # Append every record in one iteration
            output, loss = network.forward()
            train_acc = ((output - network.y) <= network.threshold_for_error).to(torch.float32).mean().cpu().detach()
            experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
            experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
            experiments_record["nb_node"].append(network.nb_node)
            experiments_record["nb_node_pruned"].append(network.nb_node_pruned)
            network.nb_node_pruned = 0


        experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
        experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

        model_experiments_record = {"network" : network, "experiments_record" : experiments_record}
    
        # inferencing
        valid_acc = evaluating.inferencing(network, x_test, y_test)
        model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

        # Plot graph
        model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

    return model_experiments_record, model_params, model_fig_drt

    # save checkpoints
    # filepath = "./checkpoints/algorithm_2/partial_data/" #'../checkpoints'
    # if os.path.isdir(filepath) == False: 
    #     os.mkdir(filepath)
    # filename = f"model_lr_{lr_goal}_seed_69_learned_315.pkl"
    # with open (f"./checkpoints/algorithm_2/partial_data/{filename}", "wb") as f:
    #     pickle.dump(model_record, f)

# model_experiments_record = main()