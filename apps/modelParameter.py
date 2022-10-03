import pandas as pd
from pathlib import Path

class ModelParameter():
    def __init__(self, **kwargs) -> None:
        
        """
        dataDirectory
        dataShape
        modelFile
        inputDimension
        hiddenNode
        outputDimension
        thresholdForError
        initializingNumber
        lossFunction
        learningGoal -> initializingLearningGoal
        learningRate
        learningRateLowerBound -> regularizingLearningRateLowerBound
        optimizer
        tuningTimes -> matchingTimes
        regularizingStrength
        """
        self.kwargs = kwargs
        
        # 以下這段完全可刪除
        self.dataDirectory = self.kwargs["dataDirectory"]
        self.dataShape = self.kwargs["dataShape"]
        self.modelFile = self.kwargs["modelFile"]
        self.inputDimension = self.kwargs["inputDimension"]
        self.hiddenNode = self.kwargs["hiddenNode"]
        self.outputDimension = self.kwargs["outputDimension"]
        self.thresholdForError =self.kwargs["thresholdForError"]
        self.initializingNumber = self.kwargs["initializingNumber"]
        self.lossFunction = self.kwargs["lossFunction"]
        self.initializingLearningGoal = self.kwargs["initializingLearningGoal"]
        self.learningRate = self.kwargs["learningRate"]
        self.regularizingLearningRateLowerBound = self.kwargs["regularizingLearningRateLowerBound"]
        self.optimizer = self.kwargs["optimizer"]
        self.matchingTimes = self.kwargs["matchingTimes"]
        self.regularizingStrength = self.kwargs["regularizingStrength"]

    def info(self):
        
        print(f'self.dataDirectory = {self.dataDirectory}')
        print(f'self.dataShape = {self.dataShape}')
        print(f'self.modelFile = {self.modelFile}')
        print(f'self.initializingNumber = {self.initializingNumber}')
        print(f'self.lossFunction = {self.lossFunction}')
        print(f'self.learningGoal = {self.learningGoal}')
        print(f'self.learningRate = {self.learningRate}')
        print(f'self.learningRateLowerBound = {self.learningRateLowerBound}')
        print(f'self.optimizer = {self.optimizer}')
        print(f'self.tuningTimes = {self.tuningTimes}')
        print(f'self.regularizingStrength = {self.regularizingStrength}')

    def get_dataShape(path):
        
        p = Path(path).glob('**/*')
        files = [str(x) for x in p if x.is_file()]
        shapes = [pd.read_csv(filePath).shape for filePath in files]
        dataShape = {"X":shapes[0], "Y":shapes[1]}

        return dataShape

class ModelParameter2LayerNet(ModelParameter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class ModelParameter2(ModelParameter):
    def __init__(self, **kwargs):

        """
        dataDirectory
        dataDescribing
        dataShape
        modelFile
        inputDimension
        hiddenNode
        outputDimension
        activationFunction
        thresholdForError
        lossFunction
        optimizer
        learningRate
        betas
        eps
        weightDecay
        initializingRule
        initializingNumber
        initializingLearningGoal
        selectingRule
        matchingRule
        matchingTimes
        matchingLearningGoal
        matchingLearningRateLowerBound
        crammingRule
        reorganizingRule
        regularizingTimes
        regularizingStrength
        regularizingLearningGoal
        regularizingLearningRateLowerBound
        """

        self.kwargs = kwargs
        
        # 以下這段完全可刪除
        self.dataDirectory = self.kwargs["dataDirectory"]
        self.dataDescribing = self.kwargs["dataDescribing"]
        self.dataShape = self.kwargs["dataShape"]
        self.modelFile = self.kwargs["modelFile"]
        self.inputDimension = self.kwargs["inputDimension"]
        self.hiddenNode = self.kwargs["hiddenNode"]
        self.outputDimension = self.kwargs["outputDimension"]
        self.activationFunction = self.kwargs["activationFunction"]
        self.thresholdForError =self.kwargs["thresholdForError"]
        self.lossFunction = self.kwargs["lossFunction"]
        self.optimizer = self.kwargs["optimizer"]
        self.learningRate = self.kwargs["learningRate"]
        self.betas = self.kwargs["betas"]
        self.eps = self.kwargs["eps"]
        self.weightDecay = self.kwargs["weightDecay"]
        self.initializingRule = self.kwargs["initializingRule"]
        self.initializingNumber = self.kwargs["initializingNumber"]
        self.initializingLearningGoal = self.kwargs["initializingLearningGoal"]
        self.selectingRule = self.kwargs["selectingRule"]
        self.matchingRule = self.kwargs["matchingRule"]
        self.matchingTimes = self.kwargs["matchingTimes"]
        self.matchingLearningGoal = self.kwargs["matchingLearningGoal"]
        self.matchingLearningRateLowerBound = self.kwargs["matchingLearningRateLowerBound"]
        self.crammingRule = self.kwargs["crammingRule"]
        self.reorganizingRule = self.kwargs["reorganizingRule"]
        self.regularizingTimes = self.kwargs["regularizingTimes"]
        self.regularizingStrength = self.kwargs["regularizingStrength"]
        self.regularizingLearningGoal = self.kwargs["regularizingLearningGoal"]
        self.regularizingLearningRateLowerBound = self.kwargs["regularizingLearningRateLowerBound"]
    
    def info(self):
        
        print(f'self.kwargs = {self.kwargs}')
    
"""
        self.model_params["activationFunction"]
        self.model_params["lossFunction"]
        self.model_params["optimizer"]
        self.model_params["learningRate"]
        self.model_params["betas"]
        self.model_params["eps"]
        self.model_params["weightDecay"]
        self.model_params["initializingRule"]
        self.model_params["initializingNumber"]
        self.model_params["initializingLearningGoal"]
        self.model_params["selectingRule"]
        self.model_params["matchingRule"]
        self.model_params["matchingTimes"]
        self.model_params["matchingLearningGoal"]
        self.model_params["matchingLearningRateLowerBound"]
        self.model_params["crammingRule"]
        self.model_params["reorganizingRule"]
        self.model_params["regularizingTimes"]
        self.model_params["regularizingStrength"]
        self.model_params["regularizingLearningGoal"]
        self.model_params["regularizingLearningRateLowerBound"]
"""