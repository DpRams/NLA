import pandas as pd
from pathlib import Path

class ModelParameter():
    def __init__(self, **kwargs) -> None:
        
        """
        ---Scenario 1 parameter---
        * dataDirectory
        * dataShape
        * modelFile
        * inputDimension
        * hiddenNode
        * outputDimension
        * thresholdForError
        * initializingNumber
        * lossFunction
        * learningGoal -> initializingLearningGoal
        * learningRate
        * learningRateLowerBound -> regularizingLearningRateLowerBound
        * optimizer
        * tuningTimes -> matchingTimes
        * regularizingStrength

        ---SLFN parameter---
        * dataDirectory
        * hiddenNode
        * activationFunction
        * epoch
        * batchSize
        * learningGoal (Demo訂在實數output，訓練時算loss用)
        * thresholdForError(準備移除) -> rmseError
        * lossFunction
        * optimizer
        * learningRate
        * betas
        * eps
        * weightDecay

        ---ASLFN parameter---
        dataDirectory
        dataDescribing
        dataShape
        modelFile
        inputDimension
        hiddenNode
        outputDimension
        activationFunction
        thresholdForError(準備移除) -> rmseError
        lossFunction
        optimizer
        learningRate
        betas
        eps
        weightDecay
        initializingRule
        initializingNumber
        learningGoal
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

    def get_dataShape(path):
        
        p = Path(path).glob('**/*')
        files = [str(x) for x in p if x.is_file()]
        shapes = [pd.read_csv(filePath).shape for filePath in files]
        dataShape = {"X":shapes[0], "Y":shapes[1]}

        return dataShape



