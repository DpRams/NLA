import yaml
import sys

availablePort = 80
ymlDict = [{"stages" : ["build", "test", "deploy"]}, \
           {"build" : {"stage" : "build", "tags" : "AILab", "only" : "scenario_2", "script" : "echo \"build\""}}, \
           {"test" : {"stage" : "test", "tags" : "AILab", "only" : "scenario_2", "script" : "echo \"testing\""}}, \
           {"deploy" : {"stage" : "deploy", "tags" : "AILab", \
                        "script" : ["echo \"deploy\"", "docker", "cd ASLFN", "docker build -t aslfn:latest -f rootuser.Dockerfile .", f"docker run -p {availablePort}:{availablePort} -d aslfn:latest"], \
                        "rules" : {"changes" : "ASLFN/docker_apps/*"}}}]

with open(".\\test.yml", 'w') as file:
    documents = yaml.dump(ymlDict, file)