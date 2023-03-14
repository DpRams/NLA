import torch 
import os 
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from apps import evaluating, saving
from network.net import Hw1
from model_file import hw1

def ensembling(selected_models, dataDirectory):

    model_list = {}
    for model in selected_models:
        ptPath = Path(f"{root}\\model_registry\\pt\\{model}")
        model_list[model] = torch.load(ptPath)

    model_experiments_record = evaluating_ensemble(model_list, dataDirectory)
    validLoss = model_experiments_record["experiments_record"]["valid"]["mean_loss"]

    return validLoss

def evaluating_ensemble(model_list, dataDirectory):

    batchSize = 10
    _, val_loader = hw1.reading_dataset_Training_only_2LayerNet(dataDirectory, batchSize)

    # Initialize record
    model_experiments_record = {"network" : None, "experiments_record" : None}
    experiments_record = {"train" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}, \
                            "valid" : {"mean_acc" : 0, "acc_step" : [], "mean_loss" : 0, "loss_step" : []}}

    loss_per_epoch = []
        
    for data in val_loader:

        x, y = data

        model_loss_list = []

        for model in model_list.values():

            model.eval()

            model.setData(x, y)
            
            with torch.no_grad():
                output, loss = model.forward()
                model_loss_list.append(loss.item())

        # 平均 loss 
        avg_loss_models = np.mean(model_loss_list)

        loss_per_epoch.append(avg_loss_models)

        experiments_record["valid"]["loss_step"].append(np.round(np.mean(loss_per_epoch), 3))
        experiments_record["valid"]["mean_loss"] = np.round(np.mean(experiments_record["valid"]["loss_step"]), 3)

    model_experiments_record = {"experiments_record" : experiments_record}

    return model_experiments_record



