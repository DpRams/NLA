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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
# sys.path.append(str(parent))
sys.path.append(str(root))

from apps import evaluating, saving
from network.net import Network


class CustomDataset(Dataset):
    
    def __init__(self, X, y) -> None:
        
        self.X = X
        self.y = y
        # convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def reading_dataset_Training_only_2LayerNet(dataDirecotry, batchSize):
    
    filelist = os.listdir(f"./upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    # build dataset
    dataset = CustomDataset(X_transformed, Y_transformed)

    # create data indices for train val split
    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(np.floor(0.2 * data_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # create data loader
    train_loader = DataLoader(dataset, batch_size=batchSize, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batchSize, sampler=val_sampler)
    
    return (train_loader, val_loader)


def training_2LayerNet(MyNet, model_params):

    train_loader, val_loader = reading_dataset_Training_only_2LayerNet(MyNet.model_params["dataDirectory"], MyNet.model_params["batchSize"])

    # Initialize record
    model_experiments_record = {"network" : None, "experiments_record" : None}
    experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                            "valid" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}}

    for i in range(MyNet.model_params["epoch"]):

        MyNet.train()

        for batch in train_loader:
            
            x, y = batch

            MyNet.setData(x, y)
            
            output, loss = MyNet.forward()
            MyNet.backward(loss)

            train_acc = ((output - MyNet.y) <= MyNet.model_params["learningGoal"]).to(torch.float32).mean().cpu().detach()
            experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
            experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
        
        for batch in val_loader:

            x, y = batch

            MyNet.setData(x, y)
            
            with torch.no_grad():
                output, loss = MyNet.forward()

                valid_acc = ((output - MyNet.y) <= MyNet.model_params["learningGoal"]).to(torch.float32).mean().cpu().detach()
                experiments_record["valid"]["acc_step"].append(np.round(valid_acc, 3))
                experiments_record["valid"]["loss_step"].append(np.round(loss.item(), 3))

        experiments_record["train"]["mean_acc"] = np.round(np.mean(experiments_record["train"]["acc_step"]), 3)
        experiments_record["train"]["mean_loss"] = np.round(np.mean(experiments_record["train"]["loss_step"]), 3)
        experiments_record["valid"]["mean_acc"] = np.round(np.mean(experiments_record["valid"]["acc_step"]), 3)
        experiments_record["valid"]["mean_loss"] = np.round(np.mean(experiments_record["valid"]["loss_step"]), 3)

    model_experiments_record = {"network" : MyNet, "experiments_record" : experiments_record}



    # # Initialize record
    # model_experiments_record = {"network" : None, "experiments_record" : None}

    # # Reading dataset
    # (x_train_scaled, y_train_scaled, x_test, y_test) = \
    # reading_dataset_Training_only_2LayerNet(MyNet.net.model_params["dataDirectory"])  

    #     # Record experiments data
    # experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
    #                         "valid" : {"mean_acc" : 0}, \
    #                         "nb_node" : [], "nb_node_pruned" : [],\
    #                         "Route" : {"Blue": 0, "Red":0, "Green":0}}

    # experiments_record["nb_node"].append(MyNet.net.linear1.weight.data.shape[0])
    
    # for i in range(1, x_train_scaled.shape[0]):
    
    #     print('-----------------------------------------------------------')
    #     print(f'訓練第幾筆資料 : {i}')

    #     current_x = x_train_scaled[:i]
    #     current_y = y_train_scaled[:i].reshape(-1, 1)

    #     print(f'current_x = {current_x.shape}')
    #     print(f'current_y = {current_y.shape}')

    #     MyNet.net.setData(current_x, current_y)

    #     # Append every record in one iteration
    #     output, loss = MyNet.net.forward()
    #     train_acc = ((output - MyNet.net.y) <= MyNet.net.model_params["initializingLearningGoal"]).to(torch.float32).mean().cpu().detach()
    #     experiments_record["train"]["acc_step"].append(np.round(train_acc, 3))
    #     experiments_record["train"]["loss_step"].append(np.round(loss.item(), 3))
    #     experiments_record["nb_node"].append(MyNet.net.nb_node)
    #     experiments_record["nb_node_pruned"].append(MyNet.net.nb_node_pruned)
    #     MyNet.net.nb_node_pruned = 0


    # experiments_record["train"]["mean_acc"] = np.mean(experiments_record["train"]["acc_step"])
    # experiments_record["train"]["mean_loss"] = np.mean(experiments_record["train"]["loss_step"])

    # model_experiments_record = {"network" : MyNet.net, "experiments_record" : experiments_record}

    # # inferencing
    # valid_acc = evaluating.inferencing(MyNet.net, x_test, y_test)
    # model_experiments_record["experiments_record"]["valid"]["mean_acc"] = np.round(valid_acc, 3)

    # # Plot graph
    # model_fig_drt = evaluating.making_figure(model_experiments_record, model_params)    

    # return model_experiments_record, model_params, model_fig_drt

    return model_experiments_record, model_params, None

def main(model_params):

    MyNet = Network(dataDirectory = model_params.kwargs["dataDirectory"], \
                    dataShape = model_params.kwargs["dataShape"], \
                    modelFile = "custoNet_SLFN.py", \
                    inputDimension = model_params.kwargs["inputDimension"], \
                    hiddenNode = model_params.kwargs["hiddenNode"], \
                    outputDimension = model_params.kwargs["outputDimension"], \
                    activationFunction = model_params.kwargs["activationFunction"], \
                    epoch = model_params.kwargs["epoch"], \
                    batchSize = model_params.kwargs["batchSize"], \
                    learningGoal = model_params.kwargs["learningGoal"], \
                    thresholdForError = model_params.kwargs["thresholdForError"], \
                    lossFunction = model_params.kwargs["lossFunction"], \
                    optimizer = model_params.kwargs["optimizer"], \
                    learningRate = model_params.kwargs["learningRate"], \
                    betas = model_params.kwargs["betas"], \
                    eps = model_params.kwargs["eps"], \
                    weightDecay = model_params.kwargs["weightDecay"])

    print(f"查看{MyNet.model_params}")

    model_experiments_record, model_params, model_fig_drt = training_2LayerNet(MyNet, model_params)

    return model_experiments_record, model_params, model_fig_drt
