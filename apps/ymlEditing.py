import yaml
import requests
import sys
import pandas as pd
from checkPort import findPortAvailable
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

runner_tags = ["laptop-ASUS"] # AILab

def deployingModelToYml(modelId):

    availablePort = findPortAvailable()
    SERVICEPORT = 8002 # fixed

    ymlDict = {"stages": ["deploy"],
               "deploy": {"stage": "deploy", "tags": runner_tags, \
                          "script": ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", \
                                    f"docker run -p {availablePort}:{SERVICEPORT} -d aslfn:latest", \
                                    f"cd {root}\\apps", "docker ps -l | findstr aslfn > dockerTmp"], \
                          "after_script":[f"python3", f"python3 {root}\\apps\\updateDeployment.py -m {modelId} -a \"deploying\""], \
                          "rules": [{"changes": ["ASLFN/docker_apps/deployTmp"]}]}}

    # ymlDict = {"stages": ["build", "test", "deploy"],
    #            "build": {"stage": "build", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"build\""]},
    #            "test": {"stage": "test", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"testing\""]},
    #            "deploy": {"stage": "deploy", "tags": ["AILab"], \
    #                       "script": ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", \
    #                                 f"docker run -p {availablePort}:{SERVICEPORT} -d aslfn:latest", \
    #                                 f"cd {root}\\apps", "docker ps -l | findstr aslfn > dockerTmp", \
    #                                 'docker rmi $(docker images -f "dangling=true" -q)'], \
    #                       "after_script":[f"python C:\\Users\\user\\rams\\projcet\\apps\\updateDeployment.py -m {modelId} -a \"deploying\""], \
    #                       "rules": [{"changes": ["ASLFN/docker_apps/deployTmp"]}]}}

    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)

def revokingModelToYml(modelId):

    containerID = requests.get(f"http://127.0.0.1:8001/model/deployments?key=modelId&value={modelId}").json()[0]["containerID"]

    ymlDict = {"stages": ["revoke"],
               "revoke": {"stage": "revoke", "tags": runner_tags, \
                          "script": ["echo \"revoke\"", "docker", \
                                    f"docker stop {containerID}", f"docker rm {containerID}", \
                                    f"python {root}\\apps\\updateDeployment.py -m {modelId} -a \"revoking\""], \
                          "rules": [{"changes": ["apps/revokeTmp"]}]
                                    }}

    # ymlDict = {"stages": ["build", "test", "revoke"],
    #            "build": {"stage": "build", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"build\""]},
    #            "test": {"stage": "test", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"testing\""]},
    #            "revoke": {"stage": "revoke", "tags": ["AILab"], \
    #                       "script": ["echo \"revoke\"", "docker", \
    #                                 f"docker stop {containerID}", f"docker rm {containerID}", \
    #                                 f"python C:\\Users\\user\\rams\\projcet\\apps\\updateDeployment.py -m {modelId} -a \"revoking\""], \
    #                       "rules": [{"changes": ["apps/revokeTmp"]}]
    #                                 }}
    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)
