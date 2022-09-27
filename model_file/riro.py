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
from network.net import Network, RIRO


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


# 存放model, experiments_record
def main(model_params):

    lr_goals = [model_params.initializingLearningGoal]
    # model_experiments_record = {"lr_goals" : {key : None for key in lr_goals}}
    model_experiments_record = {"network" : None, "experiments_record" : None}

    for lr_goal in sorted(lr_goals, reverse=True):
        
        # Reading dataset
        (initial_x, initial_y, x_train_scaled, y_train_scaled, x_test, y_test) = reading_dataset_Training(model_params.dataDirectory, model_params.initializingNumber)
        
        # Defining model
        network = RIRO( inputDimension=model_params.inputDimension, \
                        hiddenNode=model_params.hiddenNode, \
                        outputDimension=model_params.outputDimension, \
                        lossFunction=model_params.lossFunction, \
                        initializingLearningGoal=lr_goal, \
                        learningRate=model_params.learningRate, \
                        regularizingLearningRateLowerBound=model_params.regularizingLearningRateLowerBound, \
                        optimizer=model_params.optimizer,  \
                        matchingTimes=model_params.matchingTimes,  \
                        regularizingStrength=model_params.regularizingStrength, \
                        thresholdForError = model_params.thresholdForError)

        # Initializing model
        # network = initializing(network, initial_x, initial_y)
        network.initializing(initial_x, initial_y)

        # Record experiments data
        experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                              "valid" : {"mean_acc" : 0}, \
                              "nb_node" : [], "nb_node_pruned" : [],\
                              "Route" : {"Blue": 0, "Red":0, "Green":0}}

        experiments_record["nb_node"].append(network.linear1.weight.data.shape[0])
        
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

            network.setData(current_x, current_y)
            network_pre = copy.deepcopy(network)

            output, loss = network.forward()

            if torch.all(torch.abs(output - network.y) <= network.model_params["initializingLearningGoal"]):
                network.acceptable = True
                network = network.reorganizing()
                experiments_record["Route"]["Blue"] += 1

            else:
                
                if RIRO.is_initializingNumber_too_big_to_initializing(i): return "Initializing 失敗", "Initializing 失敗", "Initializing 失敗"

                network.acceptable = False
                network.matching()

                if network.acceptable:
                    network = network.reorganizing()
                    experiments_record["Route"]["Green"] += 1

                else:

                    network = copy.deepcopy(network_pre)
                    network.cramming()
                    # print("Cramming End")
                    if RIRO.is_learningGoal_too_small_to_cramming(network): return "Cramming 失敗", "Cramming 失敗", "Cramming 失敗"
                    
                    # network = reorganizing(network)
                    network = network.reorganizing()
                    experiments_record["Route"]["Red"] += 1

            # Append every record in one iteration
            output, loss = network.forward()
            train_acc = ((output - network.y) <= network.model_params["initializingLearningGoal"]).to(torch.float32).mean().cpu().detach()
            experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
            experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
            experiments_record["nb_node"].append(network.nb_node)
            experiments_record["nb_node_pruned"].append(network.nb_node_pruned)
            network.nb_node_pruned = 0


        experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
        experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

        model_experiments_record = {"network" : network, "experiments_record" : experiments_record}
    
        # inferencing
        valid_acc = evaluating.inferencing(network, x_test, y_test, validating=True)
        model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

        # Plot graph
        model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

    return model_experiments_record, model_params, model_fig_drt
