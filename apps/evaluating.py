
import torch 
import pickle
import time
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from modelParameter import ModelParameter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

from network.net import Network


def reading_dataset_Testing(dataDirecotry):
    
    # Train
    trainingFileList = os.listdir(f"{root}/upload_data/{dataDirecotry}/Train")
    trainingFile_x, trainingFile_y = sorted(trainingFileList) # ordered by prefix: X_, Y_
    trainingFilePath_X, trainingFilePath_Y = f"./upload_data/{dataDirecotry}/Train/{trainingFile_x}", f"./upload_data/{dataDirecotry}/Train/{trainingFile_y}"
    # print(f"trainingFilePath_X = {trainingFilePath_X}, trainingFilePath_Y = {trainingFilePath_Y}")
    trainingDf_X, trainingDf_Y = pd.read_csv(trainingFilePath_X), pd.read_csv(trainingFilePath_Y)

    # Test
    testingFileList = os.listdir(f"{root}/upload_data/{dataDirecotry}/Test")
    testingFile_x, testingFile_y = sorted(testingFileList) # ordered by prefix: X_, Y_
    testingFilePath_X, testingFilePath_Y = f"./upload_data/{dataDirecotry}/Test/{testingFile_x}", f"./upload_data/{dataDirecotry}/Test/{testingFile_y}"
    # print(f"testingFilePath_X = {testingFilePath_X}, testingFilePath_Y = {testingFilePath_Y}")
    testingDf_X, testingDf_Y = pd.read_csv(testingFilePath_X), pd.read_csv(testingFilePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    # Fit training data(transform is redundant)
    _ = sc_x.fit_transform(trainingDf_X.to_numpy())
    _ = sc_y.fit_transform(trainingDf_Y.to_numpy())

    # Transform testing data
    X_transformed = sc_x.transform(testingDf_X.to_numpy())
    Y_transformed = sc_y.transform(testingDf_Y.to_numpy())

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

    



# Training Accuracy, Training Loss, nb_node, nb_node_pruned 折線圖
# Route 圓餅圖
def making_figure(model_experiments_record, model_params):
    
    # this line is set to prevent the bug about https://blog.csdn.net/qq_42998120/article/details/107871863
    plt.switch_backend("agg")

    # print(f"查看{model_params}")

    data_drt = model_params.kwargs["dataDirectory"]
    
    lr_goal = model_params.kwargs["learningGoal"]

    # create dir path
    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.kwargs["modelFile"][:-3]
    validAcc = str(model_experiments_record["experiments_record"]["valid"]["mean_acc"])[:5]
    drtName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}\\" 

    # create dir    
    drtPath = Path(f"{root}\\model_fig\\{drtName}")
    drtPath.mkdir(parents=True, exist_ok=True)

    training_acc_step = model_experiments_record["experiments_record"]["train"]["acc_step"]
    training_loss_step = model_experiments_record["experiments_record"]["train"]["loss_step"]
    nb_node_step = model_experiments_record["experiments_record"]["nb_node"]
    nb_node_pruned_step = model_experiments_record["experiments_record"]["nb_node_pruned"]
    routes_cnt = model_experiments_record["experiments_record"]["Route"]

    # making figure
    __plot_acc(training_acc_step, drtPath)
    __plot_loss(training_loss_step, drtPath)
    __plot_nb_node(nb_node_step, drtPath)
    __plot_nb_node_pruned(nb_node_pruned_step, drtPath)
    __plot_routes(routes_cnt, drtPath)

    # model_fig_drt = drtPath # return the whole path
    model_fig_drt = drtPath.parts[-1] # return only the last drtPath

    return model_fig_drt


def making_figure_2LayerNet(model_experiments_record, model_params):
    
    # this line is set to prevent the bug about https://blog.csdn.net/qq_42998120/article/details/107871863
    plt.switch_backend("agg")

    # print(f"查看{model_params}")

    data_drt = model_params.kwargs["dataDirectory"]
    
    lr_goal = model_params.kwargs["learningGoal"]

    # create dir path
    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.kwargs["modelFile"][:-3]
    validAcc = str(model_experiments_record["experiments_record"]["valid"]["mean_acc"])[:5]
    drtName = f"{data_drt}_{modelType}_{lr_goal}_{validAcc}_{timeStamp}\\" 

    # create dir    
    drtPath = Path(f"{root}\\model_fig\\{drtName}")
    drtPath.mkdir(parents=True, exist_ok=True)

    training_acc_step = model_experiments_record["experiments_record"]["train"]["acc_step"]
    training_loss_step = model_experiments_record["experiments_record"]["train"]["loss_step"]
    validating_acc_step = model_experiments_record["experiments_record"]["valid"]["acc_step"]
    validating_loss_step = model_experiments_record["experiments_record"]["valid"]["loss_step"]


    # making figure
    __plot_acc(training_acc_step, drtPath)
    __plot_loss(training_loss_step, drtPath)
    __plot_acc(validating_acc_step, drtPath, validating=True)
    __plot_loss(validating_loss_step, drtPath, validating=True)

    # model_fig_drt = drtPath # return the whole path
    model_fig_drt = drtPath.parts[-1] # return only the last drtPath

    return model_fig_drt

def __plot_acc(training_acc_step, drtPath, validating=False):

    if not validating:

        plt.plot([i for i in training_acc_step], label = 'train') 
        plt.title("Training accuracy")
        plt.xlabel("Data")
        plt.ylabel("Accuracy")
        plt.grid(linestyle="--", linewidth=0.5)
        fileName = "trainingAccuracy.png"
        plt.savefig(f"{drtPath}\\{fileName}")
        plt.clf()
    else:
        plt.plot([i for i in training_acc_step], label = 'val') 
        plt.title("Validating accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(linestyle="--", linewidth=0.5)
        fileName = "validatingAccuracy.png"
        plt.savefig(f"{drtPath}\\{fileName}")
        plt.clf()


def __plot_loss(training_loss_step, drtPath, validating=False):

    if not validating:
        plt.plot([i for i in training_loss_step], label = 'train') 
        plt.title("Training loss")
        plt.xlabel("Data")
        plt.ylabel("Loss")
        plt.grid(linestyle="--", linewidth=0.5)
        fileName = "trainingLoss.png"
        plt.savefig(f"{drtPath}\\{fileName}")
        plt.clf()

    else:
        plt.plot([i for i in training_loss_step], label = 'val') 
        plt.title("Validating loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(linestyle="--", linewidth=0.5)
        fileName = "validatingLoss.png"
        plt.savefig(f"{drtPath}\\{fileName}")
        plt.clf()

def __plot_nb_node(nb_node_step, drtPath):

    plt.plot([i for i in nb_node_step], label = 'nodes') 
    plt.title("Number of nodes")
    plt.xlabel("Data")
    plt.ylabel("node")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "nodes.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()

def __plot_nb_node_pruned(nb_node_pruned_step, drtPath):
    
    plt.plot([i for i in nb_node_pruned_step], label = 'nodes_pruned') 
    plt.title("Number of pruned nodes")
    plt.xlabel("Data")
    plt.ylabel("node")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "prunedNodes.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()

def __plot_routes(routes_cnt, drtPath):

    routes_class = routes_cnt.keys()
    routes_values = list(routes_cnt.values())
    x = np.arange(len(routes_class))
    plt.bar(x, routes_values, color=['blue', 'red', 'green', 'purple'])
    plt.xticks(x, routes_class)
    plt.xlabel('Route')
    plt.ylabel('Count')
    plt.title('Route distribution')
    fileName = "routes.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()
