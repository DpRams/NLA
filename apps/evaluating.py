
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

from apps import getFreerGpu


class Network(torch.nn.Module):

    def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):

        super().__init__()

        self.setting_device()

        # Initialize
        self.linear1 = torch.nn.Linear(x_train_scaled.shape[1], nb_neuro).to(self.device)
        self.linear2 = torch.nn.Linear(nb_neuro, 1).to(self.device)
    
        # Stop criteria - threshold
        self.threshold_for_error = eval(kwargs["learning_goal"]) 
        self.threshold_for_lr = eval(kwargs["learning_rate_lower_bound"]) 
        self.tuning_times = eval(kwargs["tuning_times"]) 
        self.regularizing_strength = eval(kwargs["regularizing_strength"])
        
        # Set default now, not open for customization.
        not_used_currently = (kwargs["regularizing_strength"], kwargs["optimizer"])

        # Input data
        self.x = torch.FloatTensor(x_train_scaled).to(self.device)
        self.y = torch.FloatTensor(y_train_scaled).to(self.device)

        # Learning rate
        self.learning_rate = eval(kwargs["learning_rate"])

        # Whether the network is acceptable, default as False
        self.acceptable = False

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = nb_neuro
        
        self.undesired_index = None
        self.message = ""
    

    def setting_device(self):

        FreerGpuId = getFreerGpu.getting_freer_gpu()
        device = torch.device(f"cuda:{FreerGpuId}")
        self.device = device
        
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
    acc = (diff <= network.threshold_for_error).to(torch.float32).mean().cpu().numpy()
    acc = np.round(acc, 3)
    return acc


# Training Accuracy, Training Loss, nb_node, nb_node_pruned 折線圖
# Route 圓餅圖
def making_figure(model_experiments_record, model_params):
    
    # this line is set to prevent the bug about https://blog.csdn.net/qq_42998120/article/details/107871863
    plt.switch_backend("agg")

    data_drt = model_params.dataDirectory
    lr_goal = model_params.learningGoal

    # create dir path
    timeStamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    modelType = model_params.modelFile[:-3]
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
