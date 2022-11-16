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

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                            json={"modelId" : int(modelId), \
                                    "keyToBeChanged" : "containerID", \
                                    "valueToBeChanged" : containerID}).json()

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                        json={"modelId" : int(modelId), \
                                "keyToBeChanged" : "containerPort", \
                                "valueToBeChanged" : containerPort}).json()

def updateInfoRevoking(modelId):

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                            json={"modelId" : int(modelId), \
                                    "keyToBeChanged" : "containerID", \
                                    "valueToBeChanged" : "None"}).json()

    deployRecord = requests.put(f"http://127.0.0.1:8001/model/deployments", \
                        json={"modelId" : int(modelId), \
                                "keyToBeChanged" : "containerPort", \
                                "valueToBeChanged" : "None"}).json()

if args.action == "deploying":
    updateInfoDeploying(args.modelId)
elif args.action == "revoking":
    updateInfoRevoking(args.modelId)


