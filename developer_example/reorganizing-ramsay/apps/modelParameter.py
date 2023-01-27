import pandas as pd
from pathlib import Path

class ModelParameter():
    def __init__(self, **kwargs) -> None:
        
        """
        ---SLFN parameter---
        * dataDirectory
        * hiddenNode
        * weightInitialization
        * activationFunction
        * epoch
        * batchSize
        * learningGoal
        * testingMetric
        * lossFunction
        * optimizer
        * learningRate
        * betas
        * eps
        * weightDecay
        * timestamp

        ---ASLFN parameter---
        * dataDirectory
        * dataDescribing
        * dataShape
        * modelFile
        * inputDimension
        * hiddenNode
        * outputDimension
        * activationFunction
        * testingMetric
        * lossFunction
        * optimizer
        * learningRate
        * betas
        * eps
        * weightDecay
        * initializingRule
        * initializingNumber
        * learningGoal
        * selectingRule
        * matchingRule
        * matchingTimes
        * matchingLearningGoal
        * matchingLearningRateLowerBound
        * crammingRule
        * reorganizingRule
        * regularizingTimes
        * regularizingStrength
        * regularizingLearningGoal
        * regularizingLearningRateLowerBound
        * timestamp
        """

        self.kwargs = kwargs
        

    def info(self):
        
        print(f'self.kwargs = {self.kwargs}')

    def get_dataShape(path):
        
        p = Path(path).glob('**/*')
        files = [str(x) for x in p if x.is_file()]
        shapes = [pd.read_csv(filePath).shape for filePath in files]
        dataShape = {"X":shapes[0], "Y":shapes[1]}

        return dataShape



