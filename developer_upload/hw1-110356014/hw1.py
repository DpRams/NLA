import requests


###################################
# 以下皆為待填區域

# Step 1. 填寫下面的變數，變數請從右方註解中挑選

dataDirectory = "hospice"              # "solar", "hospice"
hiddenNode = "18"                 # any integer
weightInitialization = "xavierUniform"       # "xavierNormal", "xavierUniform", "kaimingNormal", "kaimingUniform"
activationFunction = "tanh"         # "ReLU", "tanh"
epoch = "26"                      # any integer
lossFunction = "MSE"  
regularizationTerm = "0"         # "0", "0.001", "0.0001"
optimizer = "gradientDescent"                  # "Adam", "gradientDescent", "Momentum"
learningRateDecayScheduler = "None" # "None", "Cosine"
studentId = "110356014"                  # Your student ID, e.g., "110356021"

# 以上皆為待填區域
###################################


URL = "http://140.119.19.87/pipeline/model/hw1"
data = {"dataDirectory" : dataDirectory,
        "hiddenNode" : hiddenNode,
        "weightInitialization" : weightInitialization,
        "activationFunction" : activationFunction,
        "epoch" : epoch,
        "lossFunction" : lossFunction,
        "regularizationTerm" : regularizationTerm,
        "optimizer" : optimizer,
        "learningRateDecayScheduler" : learningRateDecayScheduler,
        "studentId" : studentId
        }

res = requests.post(URL, data=data)

if res.status_code == 200:
    print("已成功建立模型，請到 http://140.119.19.87/pipeline/model/hw1/ensemble 確認模型")
else:
    print(f'錯誤碼: {res.status_code} - {res.reason}')