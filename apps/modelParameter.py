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
        initializingNumber
        lossFunction
        learningGoal
        learningRate
        learningRateLowerBound
        optimizer
        tuningTimes
        regularizingStrength
        """
        self.kwargs = kwargs
        
        self.dataDirectory = self.kwargs["dataDirectory"]
        self.dataShape = self.kwargs["dataShape"]
        self.modelFile = self.kwargs["modelFile"]
        self.inputDimension = self.kwargs["inputDimension"]
        self.hiddenNode = self.kwargs["hiddenNode"]
        self.outputDimension = self.kwargs["outputDimension"]
        self.initializingNumber = self.kwargs["initializingNumber"]
        self.lossFunction = self.kwargs["lossFunction"]
        self.learningGoal = self.kwargs["learningGoal"]
        self.learningRate = self.kwargs["learningRate"]
        self.learningRateLowerBound = self.kwargs["learningRateLowerBound"]
        self.optimizer = self.kwargs["optimizer"]
        self.tuningTimes = self.kwargs["tuningTimes"]
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

class ModelParameter2(ModelParameter):
    def __init__(self, **kwargs):

        """
        dataDirectory
        dataDescribing
        inputDimension
        hiddenNode
        outputDimension
        lossFunction
        optimizer
        learningRate
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

    def info(self):
        
        print(f'self.kwargs = {self.kwargs}')
