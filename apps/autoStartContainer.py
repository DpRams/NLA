# auto restart model container which is labeled as "deploying", due to the reason that shut down the service(docker) would not update the info in MongoDB  

import requests
import subprocess


def main():

    # start project(mongo, mongodb-python-api, nginx)
    p = subprocess.Popen(f"docker-compose up -d", shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

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
 


