import yaml
import requests
import sys
import pandas as pd
import time
from checkPort import findPortAvailable
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

runner_tags = ["AILab"] # AILab, laptop-ASUS

def deployingModelToYml(modelId):

    availablePort = findPortAvailable()
    SERVICEPORT = 8002 # fixed

    ymlDict = {"stages": ["deploy"],
               "deploy": {"stage": "deploy", "tags": runner_tags, \
                          "script": ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", \
                                    f"docker run -p {availablePort}:{SERVICEPORT} -d aslfn:latest", \
                                    f"cd {root}\\apps", "docker ps -l | findstr aslfn > dockerTmp"], \
                          "after_script":[f"echo $env:PATH", f"python {root}\\apps\\updateDeployment.py -m {modelId} -a \"deploying\""], \
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


def deployingModuleToYml(module_name, testing=True):

    module_id = module_name.split("-")[1]
    module_short = module_name[:2] # ma, cr, re
    if testing:
        module_name = f"testing-{module_short}-{module_id}" # testing-ma-ramsay

    availablePort = findPortAvailable()
    SERVICEPORT = 8005 # fixed

    ymlDict = {"stages": ["deploy"],
               "deploy": {"stage": "deploy", "tags": runner_tags, \
                          "script": ["echo \"deploy\"", "docker", f"cd developer_upload\\{module_name}", \
                                     f"tar -xf {module_name}.zip", \
                                     f"docker build -t {module_name}:latest -f rootuser.Dockerfile .", \
                                    f"docker run --name {module_name} -p {availablePort}:{SERVICEPORT} -d {module_name}:latest "], \
                          "rules": [{"changes": ["developer_upload/timeTmp"]}]}}

    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)

    with open("developer_upload\\timeTmp", 'w') as file:
        file.write(time.strftime("%y%m%d_%H%M%S", time.localtime()))
        file.close()
