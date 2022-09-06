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
from network.net import TwoLayerNet
from module.modules import Initialize, Select, Match, Cramming, Reorganize


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


def building_net(model_params):
    """
    於此處定義好，客製模型的 class ，並回傳 class(繼承外部 class)供 main function 使用
    """

    """
    # Basic data definition
    parameters : 
    dataDirectory
    dataDescribing
    """
    pass

    """
    # Two layer net & backpropagation
    parameters : 
    hiddenNode
    outputDimension
    lossFunction
    optimizer
    learningRate
    """
    only_2LayerNet = True
    if only_2LayerNet : reading_dataset_fn = reading_dataset_Training_only_2LayerNet
    else : reading_dataset_fn = reading_dataset_Training

    
    """
    # Initializing & Selecting
    parameters : 
    initializingNumber
    initializingLearningGoal
    selectingRule # POS/LTS
    """
    waitting_import_fn = None
    if model_params["selectingRule"] == "Disabled" : selecting_fn = None
    else :  selecting_fn = eval(waitting_import_fn) 

    """
    # Matching
    parameters : 
    matchingRule # EU/EU, LG/EU, LG, UA/LG, UA
    matchingTimes
    matchingLearningGoal
    matchingLearningRateLowerBound
    """
    waitting_import_fn = None
    if model_params["matchingRule"] == "Disabled" : matching_fn = None
    else :  matching_fn = eval(waitting_import_fn) 
    """
    # Cramming
    parameters : 
    crammingRule # Enabled/Disabled
    """
    waitting_import_fn = None
    if model_params["crammingRule"] == "Disabled" : cramming_fn = None
    else :  cramming_fn = eval(waitting_import_fn) 
    """
    # Reorganizing
    parameters : 
    reorganizingRule # Enabled/PCA/Disabled
    regularizingTimes
    regularizingStrength
    regularizingLearningGoal
    regularizingLearningRateLowerBound
    """
    waitting_import_fn = None
    if model_params["reorganizingRule"] == "Disabled" : reorganizing_fn = None
    else : reorganizing_fn = eval(waitting_import_fn) 

    class YourCSI():
        def __init__(self):
            self.net = yourCSI()

        def initializing(self, initial_x, initial_y):
            self.net.initializing(self)

        def selecting(self, x_train_scaled, y_train_scaled):
            self.net.selecting(self)

        def matching(self):
            self.net = self.net.matching(self)

        def cramming(self):
            self.net.cramming(self)

        def matching_reorganizing(self):
            self.net = self.net.matching_reorganizing(self)
             
        def regularizing(self):
            self.net = self.net.regularizing(self)

        def reoranizing(self):
            self.net = self.net.reoranizing(self)

    class yourCSI(TwoLayerNet):

        def __init__(self, **model_params):
            super().__init__(**model_params)

        def initializing(self, initial_x, initial_y):
            Initialize.Default(self, initial_x, initial_y)

        def selecting(self, x_train_scaled, y_train_scaled):
            sorted_index = Select.LTS(self, x_train_scaled, y_train_scaled)
            return sorted_index

        def matching(self):
            # matching_fn = "EU_LG"
            # eval(matching_fn)(self)
            return Match.EU_LG_UA(self)

        def cramming(self):
            return Cramming.ri_sro(self)

        def matching_reorganizing(self):
            return Reorganize.ri_sro(self)
             
        def regularizing(self):
            return Reorganize.regularizing(self)

        def reoranizing(self):
            return Reorganize.ri_sro(self)

    network = YourCSI()

    return network

def main(model_params):

    building_net(model_params)

    pass