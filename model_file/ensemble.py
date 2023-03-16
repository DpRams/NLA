import torch 
import os 
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from torchensemble import VotingRegressor

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
        # # 最小化 loss 
        # min_loss_models = min(model_loss_list)

        loss_per_epoch.append(avg_loss_models)

        experiments_record["valid"]["loss_step"].append(np.round(np.mean(loss_per_epoch), 3))
        experiments_record["valid"]["mean_loss"] = np.round(np.mean(experiments_record["valid"]["loss_step"]), 3)

    model_experiments_record = {"experiments_record" : experiments_record}

    return model_experiments_record




# def votingEnsembling(selected_models):

#     estimators = {}
#     for model in selected_models:
#         ptPath = Path(f"{root}\\model_registry\\pt\\{model}")
#         estimators[model] = torch.load(ptPath)

#     return list(estimators.values())

# def applyVotingRegressor(selected_models, dataDirectory):

#     estimators = votingEnsembling(selected_models)
#     model1 = estimators[0]
#     model2 = estimators[1]
#     ensemble_model = VotingRegressor(estimator=estimators, n_estimators=len(estimators))
#     batchSize = __getDatasetLength(dataDirectory)
#     train_loader, val_loader = hw1.reading_dataset_Training_only_2LayerNet(dataDirectory, batchSize)


#     ensemble_model.set_optimizer("Adam")
#     ensemble_model.set_criterion(torch.nn.functional.mse_loss)
#     ensemble_model.fit(train_loader=train_loader, test_loader=val_loader)
#     mse = ensemble_model.evaluate(test_loader=val_loader)
#     # mse = torch.nn.functional.mse_loss(y_val, y_pred)

#     return mse
        

# def __getDatasetLength(dataDirectory):

#     filelist = os.listdir(f"./upload_data/{dataDirectory}/Train")
#     file_x, file_y = sorted(filelist) # ordered by prefix: X_, Y_
#     filePath_X, filePath_Y = f"./upload_data/{dataDirectory}/Train/{file_x}", f"./upload_data/{dataDirectory}/Train/{file_y}"
#     df_X, df_Y = pd.read_csv(filePath_X), pd.read_csv(filePath_Y)

#     return len(df_X)
