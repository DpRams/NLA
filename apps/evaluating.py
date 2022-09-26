
import torch 
import pickle
import time
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from modelParameter import ModelParameter, ModelParameter2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

from network.net import Network


def reading_dataset_Testing(dataDirecotry):
    
    filelist = os.listdir(f"{root}/upload_data/{dataDirecotry}")
    file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
    filePath_X, filePath_Y = f"./upload_data/{dataDirecotry}/{file_x}", f"./upload_data/{dataDirecotry}/{file_y}"
    print(f"filePath_X = {filePath_X}\nfilePath_Y = {filePath_Y}")
    df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

    # StandardScaler
    sc_x, sc_y = StandardScaler(), StandardScaler()
    X_transformed = sc_x.fit_transform(df_X.to_numpy())
    Y_transformed = sc_y.fit_transform(df_Y.to_numpy())

    return (X_transformed, Y_transformed)

def reading_pkl(modelFile):
    
    modelPath = f"{root}/model_registry/{modelFile}"
    with open(modelPath, 'rb') as f:
        checkpoints = pickle.load(f)

    return checkpoints 



def inferencing(network, x_test, y_test):   

    network.eval()
    
    network.setData(x_test, y_test)
    output, loss = network.forward()
        
    diff = (output - network.y)
    
    if "thresholdForError" in network.model_params.keys():
        thresholdForError = network.model_params["thresholdForError"]
    else:
        thresholdForError = network.model_params["initializingLearningGoal"]

    acc = (diff <= thresholdForError).to(torch.float32).mean().cpu().numpy()
    acc = np.round(acc, 3)
    return acc


# Training Accuracy, Training Loss, nb_node, nb_node_pruned 折線圖
# Route 圓餅圖
def making_figure(model_experiments_record, model_params):
    
    # this line is set to prevent the bug about https://blog.csdn.net/qq_42998120/article/details/107871863
    plt.switch_backend("agg")

    # print(f"查看{model_params}")

    data_drt = model_params.kwargs["dataDirectory"]
    
    if "initializingLearningGoal" in model_params.kwargs.keys():
        lr_goal = model_params.kwargs["initializingLearningGoal"]
    else:
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

def __plot_acc(training_acc_step, drtPath):

    plt.plot([i.cpu().detach() for i in training_acc_step], label = 'train') 
    plt.title("Training accuracy")
    plt.xlabel("Data")
    plt.ylabel("Accuracy")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "Accuracy.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()


def __plot_loss(training_loss_step, drtPath):

    plt.plot([i for i in training_loss_step], label = 'train') 
    plt.title("Training loss")
    plt.xlabel("Data")
    plt.ylabel("Loss")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "Loss.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()

def __plot_nb_node(nb_node_step, drtPath):

    plt.plot([i for i in nb_node_step], label = 'nodes') 
    plt.title("Number of nodes")
    plt.xlabel("Data")
    plt.ylabel("node")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "Nodes.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()

def __plot_nb_node_pruned(nb_node_pruned_step, drtPath):
    
    plt.plot([i for i in nb_node_pruned_step], label = 'nodes_pruned') 
    plt.title("Number of pruned nodes")
    plt.xlabel("Data")
    plt.ylabel("node")
    plt.grid(linestyle="--", linewidth=0.5)
    fileName = "Pruned_nodes.png"
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
    fileName = "Routes.png"
    plt.savefig(f"{drtPath}\\{fileName}")
    plt.clf()
