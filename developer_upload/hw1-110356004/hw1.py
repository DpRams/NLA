import requests


###################################
# 以下皆為待填區域

# Step 1. 填寫下面的變數，變數請從右方註解中挑選

dataDirectory = "hospice"              # "solar", "hospice"

# Hidden Layers are layers of nodes between input and output layers(Ref). The hidden layer node defined here is a hyperparameter for training model.
hiddenNode = "18"                 # any integer

# Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model.
weightInitialization = "xavierUniform"       # "xavierNormal", "xavierUniform", "kaimingNormal", "kaimingUniform"

# In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs. There are typical activation functions, including Rectified linear unit(ReLU), sigmoid, Hyperbolic tangent(tanh).
activationFunction = "tanh"         # "ReLU", "tanh"

#The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset.
epoch = "26"                      # any integer

# At its core, a loss function is incredibly simple: It’s a method of evaluating how well your algorithm models your dataset. If your predictions are totally off, your loss function will output a higher number. If they’re pretty good, it’ll output a lower number.
lossFunction = "MSE"  
regularizationTerm = "0"         # "0", "0.001", "0.0001"

# An optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rate. Thus, it helps in reducing the overall loss and improve the accuracy.
optimizer = "gradientDescent"                  # "Adam", "gradientDescent", "Momentum"

# Learning rate decay scheduler is a technique used in machine learning to adjust the learning rate during training. A learning rate decay scheduler adjusts the learning rate according to a predefined schedule. The idea is to start with a high learning rate and gradually decrease it over time, so the model can make large updates to the weights at the beginning of training and then make smaller updates as it gets closer to the optimal solution.
learningRateDecayScheduler = "None" # "None", "Cosine"

# Your student ID
studentId = "110356004"          # e.g., "110356021"

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