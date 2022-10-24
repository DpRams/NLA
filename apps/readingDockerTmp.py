

def getContainerIDPort():
    with open(".\\apps\\dockerTmp", "r", encoding="utf-16") as file:
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

# containerID, containerPort = getContainerIDPort()
# print(containerID, containerPort)