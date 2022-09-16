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


# def building_net(model_params):
#     """
#     於此處定義好，客製模型的 class ，並回傳 class(繼承外部 class)供 main function 使用
#     """

#     """
#     # Basic data definition
#     parameters : 
#     dataDirectory
#     dataDescribing
#     """
#     pass

#     """
#     # Two layer net & backpropagation
#     parameters : 
#     hiddenNode
#     outputDimension
#     lossFunction
#     optimizer
#     learningRate
#     """
#     only_2LayerNet = True
#     if only_2LayerNet : reading_dataset_fn = reading_dataset_Training_only_2LayerNet
#     else : reading_dataset_fn = reading_dataset_Training

    
#     """
#     # Initializing & Selecting
#     parameters : 
#     initializingNumber
#     initializingLearningGoal
#     selectingRule # POS/LTS
#     """
#     waitting_import_fn = None
#     if model_params["selectingRule"] == "Disabled" : selecting_fn = None
#     else :  selecting_fn = eval(waitting_import_fn) 

#     """
#     # Matching
#     parameters : 
#     matchingRule # EU/EU, LG/EU, LG, UA/LG, UA
#     matchingTimes
#     matchingLearningGoal
#     matchingLearningRateLowerBound
#     """
#     waitting_import_fn = None
#     if model_params["matchingRule"] == "Disabled" : matching_fn = None
#     else :  matching_fn = eval(waitting_import_fn) 
#     """
#     # Cramming
#     parameters : 
#     crammingRule # Enabled/Disabled
#     """
#     waitting_import_fn = None
#     if model_params["crammingRule"] == "Disabled" : cramming_fn = None
#     else :  cramming_fn = eval(waitting_import_fn) 
#     """
#     # Reorganizing
#     parameters : 
#     reorganizingRule # Enabled/PCA/Disabled
#     regularizingTimes
#     regularizingStrength
#     regularizingLearningGoal
#     regularizingLearningRateLowerBound
#     """
#     waitting_import_fn = None
#     if model_params["reorganizingRule"] == "Disabled" : reorganizing_fn = None
#     else : reorganizing_fn = eval(waitting_import_fn) 

#     class YourCSI():
#         def __init__(self):
#             self.net = yourCSI()

#         def initializing(self, initial_x, initial_y):
#             self.net.initializing(self, initial_x, initial_y)

#         def selecting(self, x_train_scaled, y_train_scaled):
#             self.net.selecting(self, x_train_scaled, y_train_scaled)

#         def matching(self):
#             self.net = self.net.matching(self)

#         def cramming(self):
#             self.net.cramming(self)

#         def matching_reorganizing(self):
#             self.net = self.net.matching_reorganizing(self)
             
#         def regularizing(self):
#             self.net = self.net.regularizing(self)

#         def reoranizing(self):
#             self.net = self.net.reoranizing(self)

#     class yourCSI(TwoLayerNet):

#         def __init__(self, **model_params):
#             super().__init__(**model_params)

#         def initializing(self, initial_x, initial_y):
#             Initialize.Default(self, initial_x, initial_y)

#         def selecting(self, x_train_scaled, y_train_scaled):
#             sorted_index = Select.LTS(self, x_train_scaled, y_train_scaled)
#             return sorted_index

#         def matching(self):
#             # matching_fn = "EU_LG"
#             # eval(matching_fn)(self)
#             return Match.EU_LG_UA(self)

#         def cramming(self):
#             return Cram.ri_sro(self)

#         def matching_reorganizing(self):
#             return Reorganize.ri_sro(self)
             
#         def regularizing(self):
#             return Reorganize.regularizing(self)

#         def reoranizing(self):
#             return Reorganize.ri_sro(self)

#     network = YourCSI()

#     return network

def main(model_params):

    lr_goals = [model_params.initializingLearningGoal]
    # model_experiments_record = {"lr_goals" : {key : None for key in lr_goals}}
    model_experiments_record = {"network" : None, "experiments_record" : None}

    for lr_goal in sorted(lr_goals, reverse=True):
        
        # Reading dataset
        (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test) = reading_dataset_Training(model_params.dataDirectory, model_params.initializingNumber)
        
        # Defining model # modelFile
        network = YourCSI_s2(   inputDimension = model_params.inputDimension, \
                                hiddenNode = model_params.hiddenNode, \
                                outputDimension = model_params.outputDimension, \
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

        # Initializing model
        # network = initializing(network, initial_x, initial_y)
        network.initializing(initial_x, initial_y)

        # Record experiments data
        experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                              "valid" : {"mean_acc" : 0}, \
                              "nb_node" : [], "nb_node_pruned" : [],\
                              "Route" : {"Blue": 0, "Red":0, "Green":0}}

        experiments_record["nb_node"].append(network.net.linear1.weight.data.shape[0])
        
        # The initializing-use data should be add into the training data too.
        current_x, current_y = initial_x, initial_y 

        for i in range(1, x_train_scaled.shape[0]):
        
            print('-----------------------------------------------------------')
            print(f"訓練第幾筆資料 : {i + model_params.initializingNumber}")

            sorted_index = network.selecting(x_train_scaled[i-1:], y_train_scaled[i-1:])
            current_x = np.append(current_x, x_train_scaled[sorted_index[0]]).reshape(-1, x_train_scaled.shape[1])
            current_y = np.append(current_y, y_train_scaled[sorted_index[0]].reshape(-1, 1))
            current_y = np.expand_dims(current_y, 1) #turn shape [n] into [n,1] 

            print(f'current_x = {current_x.shape}')
            print(f'current_y = {current_y.shape}')

            network.net.setData(current_x, current_y)
            network_pre = copy.deepcopy(network)

            output, loss = network.net.forward()

            if torch.all(torch.abs(output - network.net.y) <= network.net.threshold_for_error):
                network.net.acceptable = True
                network.reorganizing()
                experiments_record["Route"]["Blue"] += 1

            else:
                
                if RIRO.is_initializingNumber_too_big_to_initializing(i): return "Initializing 失敗", "Initializing 失敗", "Initializing 失敗"

                network.acceptable = False
                network.matching()

                if network.acceptable:
                    network.reorganizing()
                    experiments_record["Route"]["Green"] += 1

                else:

                    network = copy.deepcopy(network_pre)
                    network.cramming()
                    # print("Cramming End")
                    if RIRO.is_learningGoal_too_small_to_cramming(network): return "Cramming 失敗", "Cramming 失敗", "Cramming 失敗"
                    
                    # network = reorganizing(network)
                    network.reorganizing()
                    experiments_record["Route"]["Red"] += 1

            # Append every record in one iteration
            output, loss = network.net.forward()
            train_acc = ((output - network.net.y) <= network.net.threshold_for_error).to(torch.float32).mean().cpu().detach()
            experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
            experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
            experiments_record["nb_node"].append(network.net.nb_node)
            experiments_record["nb_node_pruned"].append(network.net.nb_node_pruned)
            network.net.nb_node_pruned = 0


        experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
        experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

        model_experiments_record = {"network" : network.net, "experiments_record" : experiments_record}
    
        # inferencing
        valid_acc = evaluating.inferencing(network.net, x_test, y_test)
        model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

        # Plot graph
        model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

    return model_experiments_record, model_params, model_fig_drt