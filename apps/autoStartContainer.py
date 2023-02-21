# auto restart model container which is labeled as "deploying", due to the reason that shut down the service(docker) would not update the info in MongoDB  

import requests
import subprocess
import time
import readingDockerTmp


def main():

    # create volume
    p = subprocess.Popen(f"docker volume create --name=mongodbdata", shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    # start project(mongo, mongodb-python-api, nginx)
    p = subprocess.Popen(f"docker-compose up -d", shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    # start developer_modules_container
    developer_matching = readingDockerTmp.getModulesOnDocker(module_kind="matching")["module_name"]
    developer_cramming = readingDockerTmp.getModulesOnDocker(module_kind="cramming")["module_name"]
    developer_reorganizing = readingDockerTmp.getModulesOnDocker(module_kind="reorganizing")["module_name"]

    modules_to_start = f"{' '.join(developer_matching)} {' '.join(developer_cramming)} {' '.join(developer_reorganizing)}"

    p = subprocess.Popen(f"docker start {modules_to_start}", shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    # time.sleep(5)

    # get ASLFN container ID from mongo
    if requests.get(f"http://127.0.0.1:8001/model/deployments/counts").json() != 0:
        containerToBeRestart = (" ").join([model["containerID"] for model in requests.get(f"http://127.0.0.1:8001/model/deployments?key=deployStatus&value=deploying").json()])
        # start ASLFN container
        if containerToBeRestart != "":
            p = subprocess.Popen(f"docker start {containerToBeRestart}", shell=True, stdout=subprocess.PIPE)
            stdout, stderr = p.communicate()
            # print(stdout, stderr)
        else:
            return
    else:
        return
 


