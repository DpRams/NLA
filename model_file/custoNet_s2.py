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
from network.net import RIRO, YourCSI_s2


"""
與riro同等層級，用作training。
"""


def reading_dataset_Training(dataDirecotry, initializingNumber):
    
    filelist = os.listdir(f"./upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    # train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, Y_transformed, test_size=0.2, random_state=42)

    # Split data into intializing use and training use.
    initial_x, x_train_scaled = torch.FloatTensor(x_train[:initializingNumber]), torch.FloatTensor(x_train[initializingNumber:])
    initial_y, y_train_scaled = torch.FloatTensor(y_train[:initializingNumber]), torch.FloatTensor(y_train[initializingNumber:])
    
    return (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test)

def reading_dataset_Training_only_2LayerNet(dataDirecotry):
    
    filelist = os.listdir(f"./upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    # train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, Y_transformed, test_size=0.2, random_state=42)
    
    return (x_train, y_train, x_test, y_test)


def training_2LayerNet(MyCSI, model_params):

    # Initialize record
    model_experiments_record = {"network" : None, "experiments_record" : None}

    # Reading dataset
    (x_train_scaled, y_train_scaled, x_test, y_test) = \
    reading_dataset_Training_only_2LayerNet(MyCSI.net.model_params["dataDirectory"])  

        # Record experiments data
    experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                            "valid" : {"mean_acc" : 0}, \
                            "nb_node" : [], "nb_node_pruned" : [],\
                            "Route" : {"Blue": 0, "Red":0, "Green":0}}

    experiments_record["nb_node"].append(MyCSI.net.linear1.weight.data.shape[0])
    
    for i in range(1, x_train_scaled.shape[0]):
    
        print('-----------------------------------------------------------')
        print(f'訓練第幾筆資料 : {i}')

        current_x = x_train_scaled[:i]
        current_y = y_train_scaled[:i].reshape(-1, 1)

        print(f'current_x = {current_x.shape}')
        print(f'current_y = {current_y.shape}')

        MyCSI.net.setData(current_x, current_y)

        # Append every record in one iteration
        output, loss = MyCSI.net.forward()
        train_acc = ((output - MyCSI.net.y) <= MyCSI.net.model_params["initializingLearningGoal"]).to(torch.float32).mean().cpu().detach()
        experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
        experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
        experiments_record["nb_node"].append(MyCSI.net.nb_node)
        experiments_record["nb_node_pruned"].append(MyCSI.net.nb_node_pruned)
        MyCSI.net.nb_node_pruned = 0


    experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
    experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

    model_experiments_record = {"network" : MyCSI.net, "experiments_record" : experiments_record}

    # inferencing
    valid_acc = evaluating.inferencing(MyCSI.net, x_test, y_test)
    model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

    # Plot graph
    model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

    return model_experiments_record, model_params, model_fig_drt

def training_CSINet(MyCSI, model_params):
    
    # Initialize record
    model_experiments_record = {"network" : None, "experiments_record" : None}

    # print(f"查看 MyCSI.net.model_params {MyCSI.net.model_params}")

    # Reading dataset
    (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test) = \
    reading_dataset_Training(MyCSI.net.model_params["dataDirectory"], MyCSI.net.model_params["initializingNumber"])  


    # Initializing model
    MyCSI.initializing(initial_x, initial_y)

    # Record experiments data
    experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                            "valid" : {"mean_acc" : 0}, \
                            "nb_node" : [], "nb_node_pruned" : [],\
                            "Route" : {"Blue": 0, "Red":0, "Green":0}}

    experiments_record["nb_node"].append(MyCSI.net.linear1.weight.data.shape[0])
    
    # The initializing-use data should be add into the training data too.
    current_x, current_y = initial_x, initial_y 

    for i in range(1, x_train_scaled.shape[0]):
    
        print('-----------------------------------------------------------')
        print(f'訓練第幾筆資料 : {i + MyCSI.net.model_params["initializingNumber"]}')


        sorted_index = MyCSI.selecting(x_train_scaled[i-1:], y_train_scaled[i-1:])
        current_x = np.append(current_x, x_train_scaled[sorted_index[0]]).reshape(-1, x_train_scaled.shape[1])
        current_y = np.append(current_y, y_train_scaled[sorted_index[0]].reshape(-1, 1))
        current_y = np.expand_dims(current_y, 1) #turn shape [n] into [n,1] 

        print(f'current_x = {current_x.shape}')
        print(f'current_y = {current_y.shape}')

        MyCSI.net.setData(current_x, current_y)
        MyCSI_pre = copy.deepcopy(MyCSI)

        output, loss = MyCSI.net.forward()

        if torch.all(torch.abs(output - MyCSI.net.y) <= MyCSI.net.model_params["initializingLearningGoal"]):
            MyCSI.net.acceptable = True
            MyCSI.reorganizing()
            experiments_record["Route"]["Blue"] += 1

        else:
            
            if RIRO.is_initializingNumber_too_big_to_initializing(i): return "Initializing 失敗", "Initializing 失敗", "Initializing 失敗"

            MyCSI.net.acceptable = False
            MyCSI.matching()

            if MyCSI.net.acceptable:
                MyCSI.reorganizing()
                experiments_record["Route"]["Green"] += 1

            else:

                MyCSI = copy.deepcopy(MyCSI_pre)
                MyCSI.cramming()
                # print("Cramming End")
                if RIRO.is_learningGoal_too_small_to_cramming(MyCSI): return "Cramming 失敗", "Cramming 失敗", "Cramming 失敗"
                
                # network = reorganizing(network)
                MyCSI.reorganizing()
                experiments_record["Route"]["Red"] += 1

        # Append every record in one iteration
        output, loss = MyCSI.net.forward()
        train_acc = ((output - MyCSI.net.y) <= MyCSI.net.model_params["initializingLearningGoal"]).to(torch.float32).mean().cpu().detach()
        experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
        experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
        experiments_record["nb_node"].append(MyCSI.net.nb_node)
        experiments_record["nb_node_pruned"].append(MyCSI.net.nb_node_pruned)
        MyCSI.net.nb_node_pruned = 0


    experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
    experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

    model_experiments_record = {"network" : MyCSI.net, "experiments_record" : experiments_record}

    # inferencing
    valid_acc = evaluating.inferencing(MyCSI.net, x_test, y_test, validating=True)
    model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

    # Plot graph
    model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    
    
    return model_experiments_record, model_params, model_fig_drt



def main(model_params):

    MyCSI = YourCSI_s2(dataDirectory = model_params.dataDirectory, \
                        dataDescribing = model_params.dataDescribing, \
                        dataShape = model_params.dataShape, \
                        modelFile = model_params.modelFile, \
                        inputDimension = model_params.inputDimension, \
                        hiddenNode = model_params.hiddenNode, \
                        outputDimension = model_params.outputDimension, \
                        thresholdForError = model_params.thresholdForError, \
                        activationFunction = model_params.activationFunction, \
                        lossFunction = model_params.lossFunction, \
                        optimizer = model_params.optimizer, \
                        learningRate = model_params.learningRate, \
                        betas = model_params.betas, \
                        eps = model_params.eps, \
                        weightDecay = model_params.weightDecay, \
                        initializingRule = model_params.initializingRule, \
                        initializingNumber = model_params.initializingNumber, \
                        initializingLearningGoal = model_params.initializingLearningGoal, \
                        selectingRule = model_params.selectingRule, \
                        matchingRule = model_params.matchingRule, \
                        matchingTimes = model_params.matchingTimes, \
                        matchingLearningGoal = model_params.matchingLearningGoal, \
                        matchingLearningRateLowerBound = model_params.matchingLearningRateLowerBound, \
                        crammingRule = model_params.crammingRule, \
                        reorganizingRule = model_params.reorganizingRule, \
                        regularizingTimes = model_params.regularizingTimes, \
                        regularizingStrength = model_params.regularizingStrength, \
                        regularizingLearningGoal = model_params.regularizingLearningGoal, \
                        regularizingLearningRateLowerBound = model_params.regularizingLearningRateLowerBound)

    print(f"查看{MyCSI.net.model_params}")

    if MyCSI.net.model_params["initializingRule"] == "Disabled":
        training_2LayerNet(MyCSI, model_params)
    elif MyCSI.net.model_params["initializingRule"] == "LinearRegression":
        # 其實不需要把 model_params 再傳進去，這裡遇到的問題是，riro, custoNet_s1 其實在 making_figure() 輸入 ModelParameter 物件
        # 但此處 如果是輸入 MyCSI.net.model_params 則變成 "字典"，因此需要再做統整(物件比較好)。
        model_experiments_record, model_params, model_fig_drt = training_CSINet(MyCSI, model_params) 

    return model_experiments_record, model_params, model_fig_drt

# def main(model_params):

#     lr_goals = [model_params.initializingLearningGoal]
#     # model_experiments_record = {"lr_goals" : {key : None for key in lr_goals}}
#     model_experiments_record = {"network" : None, "experiments_record" : None}

#     for lr_goal in sorted(lr_goals, reverse=True):
        
#         # Reading dataset
#         (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test) = reading_dataset_Training(model_params.dataDirectory, model_params.initializingNumber)
        
#         # Defining model # modelFile
#         network = YourCSI_s2(   inputDimension = model_params.inputDimension, \
#                                 hiddenNode = model_params.hiddenNode, \
#                                 outputDimension = model_params.outputDimension, \
#                                 activationFunction = model_params.activationFunction, \
#                                 lossFunction = model_params.lossFunction, \
#                                 optimizer = model_params.optimizer, \
#                                 learningRate = model_params.learningRate, \
#                                 betas = model_params.betas, \
#                                 eps = model_params.eps, \
#                                 weightDecay = model_params.weightDecay, \
#                                 initializingRule = model_params.initializingRule, \
#                                 initializingNumber = model_params.initializingNumber, \
#                                 initializingLearningGoal = model_params.initializingLearningGoal, \
#                                 selectingRule = model_params.selectingRule, \
#                                 matchingRule = model_params.matchingRule, \
#                                 matchingTimes = model_params.matchingTimes, \
#                                 matchingLearningGoal = model_params.matchingLearningGoal, \
#                                 matchingLearningRateLowerBound = model_params.matchingLearningRateLowerBound, \
#                                 crammingRule = model_params.crammingRule, \
#                                 reorganizingRule = model_params.reorganizingRule, \
#                                 regularizingTimes = model_params.regularizingTimes, \
#                                 regularizingStrength = model_params.regularizingStrength, \
#                                 regularizingLearningGoal = model_params.regularizingLearningGoal, \
#                                 regularizingLearningRateLowerBound = model_params.regularizingLearningRateLowerBound)

#         # Initializing model
#         # network = initializing(network, initial_x, initial_y)
#         network.initializing(initial_x, initial_y)

#         # Record experiments data
#         experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
#                               "valid" : {"mean_acc" : 0}, \
#                               "nb_node" : [], "nb_node_pruned" : [],\
#                               "Route" : {"Blue": 0, "Red":0, "Green":0}}

#         experiments_record["nb_node"].append(network.net.linear1.weight.data.shape[0])
        
#         # The initializing-use data should be add into the training data too.
#         current_x, current_y = initial_x, initial_y 

#         for i in range(1, x_train_scaled.shape[0]):
        
#             print('-----------------------------------------------------------')
#             print(f"訓練第幾筆資料 : {i + model_params.initializingNumber}")

#             sorted_index = network.selecting(x_train_scaled[i-1:], y_train_scaled[i-1:])
#             current_x = np.append(current_x, x_train_scaled[sorted_index[0]]).reshape(-1, x_train_scaled.shape[1])
#             current_y = np.append(current_y, y_train_scaled[sorted_index[0]].reshape(-1, 1))
#             current_y = np.expand_dims(current_y, 1) #turn shape [n] into [n,1] 

#             print(f'current_x = {current_x.shape}')
#             print(f'current_y = {current_y.shape}')

#             network.net.setData(current_x, current_y)
#             network_pre = copy.deepcopy(network)

#             output, loss = network.net.forward()

#             if torch.all(torch.abs(output - network.net.y) <= network.net.threshold_for_error):
#                 network.net.acceptable = True
#                 network.reorganizing()
#                 experiments_record["Route"]["Blue"] += 1

#             else:
                
#                 if RIRO.is_initializingNumber_too_big_to_initializing(i): return "Initializing 失敗", "Initializing 失敗", "Initializing 失敗"

#                 network.acceptable = False
#                 network.matching()

#                 if network.acceptable:
#                     network.reorganizing()
#                     experiments_record["Route"]["Green"] += 1

#                 else:

#                     network = copy.deepcopy(network_pre)
#                     network.cramming()
#                     # print("Cramming End")
#                     if RIRO.is_learningGoal_too_small_to_cramming(network): return "Cramming 失敗", "Cramming 失敗", "Cramming 失敗"
                    
#                     # network = reorganizing(network)
#                     network.reorganizing()
#                     experiments_record["Route"]["Red"] += 1

#             # Append every record in one iteration
#             output, loss = network.net.forward()
#             train_acc = ((output - network.net.y) <= network.net.threshold_for_error).to(torch.float32).mean().cpu().detach()
#             experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
#             experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
#             experiments_record["nb_node"].append(network.net.nb_node)
#             experiments_record["nb_node_pruned"].append(network.net.nb_node_pruned)
#             network.net.nb_node_pruned = 0


#         experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
#         experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

#         model_experiments_record = {"network" : network.net, "experiments_record" : experiments_record}
    
#         # inferencing
#         valid_acc = evaluating.inferencing(network.net, x_test, y_test)
#         model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

#         # Plot graph
#         model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

#     return model_experiments_record, model_params, model_fig_drt