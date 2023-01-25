import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]

def getContainerIDPort():
    with open(f"{root}\\apps\\dockerTmp", "r", encoding="utf-16") as file:
        dockerTmp = file.read()
        dockerTmp = dockerTmp.split("   ")
        containerID = dockerTmp[0]
        containerPort = getPortFromString(dockerTmp[5])
        return containerID, containerPort

def getPortFromString(string):
    """
    0.0.0.0:8360->8002/tcp
    """
    leftIdx = string.find(":")
    rightIdx = string.find("-")
    return string[leftIdx+1:rightIdx]


def getModulesOnDocker(module_name=None, module_kind=None):
    """
    module_name such as "matching-ramsay", "cramming-ramsay"
    module_kind such as "matching", "cramming", "reorganizing"
    """
    assert module_name == None or module_kind == None, "輸入參數請擇一，輸入 module_name 尋找一特定 module；輸入 module_kind 尋找多數類似 module。"
    assert not (module_name == None and module_kind == None), "輸入參數請擇一，輸入 module_name 尋找一特定 module；輸入 module_kind 尋找多數類似 module。"

    # 之後可能會遇到，同時多人在複寫檔案的情況，要注意。
    if module_name is not None:
        p = subprocess.Popen(f"docker ps -a | findstr {module_name} > ./apps/module_name_OnDockerTmp", shell=True, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        modules_dict = {"module_name":None, "container_port":None}

        with open(f"apps\\module_name_OnDockerTmp", "r", encoding="utf-8") as file:
            dockerTmp = file.read()
            dockerTmp = dockerTmp.replace("\n", "").split("   ")
            module_name = dockerTmp[-1].lstrip(" ")
            containerPort = getPortFromString(dockerTmp[-2])
            modules_dict["module_name"] = module_name
            modules_dict["container_port"] = containerPort
            return modules_dict

    elif module_kind is not None:
        p = subprocess.Popen(f"docker ps -a | findstr {module_kind} > ./apps/module_kind_OnDockerTmp", shell=True, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        modules_dict = {"module_name":[]}

        with open(f"{root}\\apps\\module_kind_OnDockerTmp", "r", encoding="utf-8") as file:
            for line in file.readlines():
                dockerTmp = line
                dockerTmp = dockerTmp.replace("\n", "").split("   ")
                module_name = dockerTmp[-1].lstrip(" ")
                modules_dict["module_name"].append(module_name)
            return modules_dict
