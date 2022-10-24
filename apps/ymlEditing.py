import yaml
import sys
import pandas as pd
from checkPort import findPortAvailable
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

def deployingModelToYml(modelId):

    availablePort = findPortAvailable()
    SERVICEPORT = 8002 # fixed

    ymlDict = {"stages": ["build", "test", "deploy"],
               "build": {"stage": "build", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"build\""]},
               "test": {"stage": "test", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"testing\""]},
               "deploy": {"stage": "deploy", "tags": ["AILab"], \
                          "script": ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", \
                                    f"docker run -p {availablePort}:{SERVICEPORT} -d aslfn:latest", \
                                    f"cd {root}\\apps", "docker ps -l | findstr aslfn > dockerTmp"], \
                          "after_script":[f"python C:\\Users\\user\\rams\\projcet\\apps\\updateDeploymentCsv.py -m {modelId} -a \"deploying\""], \
                          "rules": [{"changes": ["ASLFN/docker_apps/*"]}]}}

    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)

def revokingModelToYml(modelId):

    containerID = pd.read_csv(f'{root}\\model_deploying\\deployment.csv').iloc[int(modelId), 4]

    ymlDict = {"stages": ["build", "test", "revoke"],
               "build": {"stage": "build", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"build\""]},
               "test": {"stage": "test", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"testing\""]},
               "revoke": {"stage": "revoke", "tags": ["AILab"], \
                          "script": ["echo \"revoke\"", "docker", \
                                    f"docker stop {containerID}", f"docker rm {containerID}", \
                                    f"python C:\\Users\\user\\rams\\projcet\\apps\\updateDeploymentCsv.py -m {modelId} -a \"revoking\""], \
                          "rules": [{"changes": ["apps/revokeTmp"]}]
                                    }}

    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)
