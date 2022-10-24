import pandas as pd
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

from readingDockerTmp import getContainerIDPort
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelId", help = "model ID")
args = parser.parse_args()

def updateInfo(modelId):

    containerID, containerPort = getContainerIDPort()

    deployRecord = pd.read_csv(f'{root}\\model_deploying\\deployment.csv')
    deployRecord.iloc[int(modelId), 4] = containerID
    deployRecord.iloc[int(modelId), 5] = containerPort
    deployRecord.to_csv(f"{root}\\model_deploying\\deployment.csv", index=None)

updateInfo(args.modelId)


