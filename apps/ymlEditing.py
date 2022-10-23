import yaml
import sys

ymlStr = """
stages:
    - build
    - test
    - deploy

build:
  stage: build
  tags:
    - "AILab"
  only:
    - scenario_2
  script:
    - echo "build"

test:
  stage: test
  tags:
    - "AILab"
  only:
    - scenario_2
  script:
    - echo "testing"

deploy:
  stage: deploy
  tags:
    - "AILab"
  script:
    - echo "deploy"
    - docker
    - cd ASLFN
    - docker build -t aslfn:latest -f rootuser.Dockerfile .
    - docker run -p 8002:8002 -d aslfn:latest
  rules:
    - changes:
      - ASLFN/docker_apps/*
"""

ymlDict = [{"stages" : ["build", "test", "deploy"]}, \
           {"build" : {"stage" : "build", "tags" : "AILab"}}]
# code = yaml.load(ymlDict, Loader=yaml.Loader)
# code["deploy"]["script"] = """
# - echo \"deploy\"
# - docker
# - cd ASLFN
# - docker build -t aslfn:latest -f rootuser.Dockerfile .
# - docker run -p 8003:8003 -d aslfn:latest"""
# code["build"]["stage"] = "test"
with open("\\test.yml", 'w') as file:
    documents = yaml.dump(ymlDict, file)