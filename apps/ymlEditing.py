import yaml
import sys
from checkPort import findPortAvailable
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

def findPortAvailableToYml(modelId):

    availablePort = findPortAvailable()
    SERVICEPORT = 8002 # fixed

    ymlDict = {"stages": ["build", "test", "deploy"],
               "build": {"stage": "build", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"build\""]},
               "test": {"stage": "test", "tags": ["AILab"], "only": ["scenario_2"], "script": ["echo \"testing\""]},
               "deploy": {"stage": "deploy", "tags": ["AILab"], \
                          "script": ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", \
                                    f"docker run -p {availablePort}:{SERVICEPORT} -d aslfn:latest", \
                                    f"cd {root}\\apps", "docker ps -l | findstr aslfn > dockerTmp"], \
                          "after_script":[f"python updateDeploymentCsv.py -m {modelId}"], \
                          "rules": [{"changes": ["ASLFN/docker_apps/*"]}]}}

    with open(".\\.gitlab-ci.yml", 'w') as file:
        yaml.dump(ymlDict, file)

# with open(".\\.gitlab-ci_org.yml", 'r') as file:
#     documents = yaml.load(ymlDict, Loader=yaml.Loader)
#     print(documents)
