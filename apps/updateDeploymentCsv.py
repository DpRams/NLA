import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parents[1]

from readingDockerTmp import getContainerIDPort
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelId", help = "model ID")
parser.add_argument("-a", "--action", help = "deploying or revoking")
args = parser.parse_args()

def updateInfoDeploying(modelId):

    containerID, containerPort = getContainerIDPort()

    deployRecord = pd.read_csv(f'{root}\\model_deploying\\deployment.csv')
    deployRecord.iloc[int(modelId), 4] = containerID
    deployRecord.iloc[int(modelId), 5] = containerPort
    deployRecord.to_csv(f"{root}\\model_deploying\\deployment.csv", index=None)

def updateInfoRevoking(modelId):

    deployRecord = pd.read_csv(f'{root}\\model_deploying\\deployment.csv')
    deployRecord.iloc[int(modelId), 4] = "None"
    deployRecord.iloc[int(modelId), 5] = "None"
    deployRecord.to_csv(f"{root}\\model_deploying\\deployment.csv", index=None)

if args.action == "deploying":
    updateInfoDeploying(args.modelId)
elif args.action == "revoking":
    updateInfoRevoking(args.modelId)


