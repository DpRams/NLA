import pandas as pd
import requests
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

    print("Before MongoDB update")

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                            json={"modelId" : int(modelId), \
                                    "keyToBeChanged" : "containerID", \
                                    "valueToBeChanged" : containerID}).json()
    print(deployRecord)

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                        json={"modelId" : int(modelId), \
                                "keyToBeChanged" : "containerPort", \
                                "valueToBeChanged" : containerPort}).json()

    print(deployRecord)
    

    print("After MongoDB update")

def updateInfoRevoking(modelId):

    deployRecord = pd.read_csv(f'{root}\\model_deploying\\deployment.csv')
    deployRecord.iloc[int(modelId), 4] = "None"
    deployRecord.iloc[int(modelId), 5] = "None"
    deployRecord.to_csv(f"{root}\\model_deploying\\deployment.csv", index=None)

    print("Before MongoDB update")
    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                            json={"modelId" : modelId, \
                                    "keyToBeChanged" : "containerID", \
                                    "valueToBeChanged" : "None"}).json()
    print(deployRecord)                                

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                        json={"modelId" : modelId, \
                                "keyToBeChanged" : "containerPort", \
                                "valueToBeChanged" : "None"}).json()
    print(deployRecord)
    print("After MongoDB update")

if args.action == "deploying":
    updateInfoDeploying(args.modelId)
elif args.action == "revoking":
    updateInfoRevoking(args.modelId)


