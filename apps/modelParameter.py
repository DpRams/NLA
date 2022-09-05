class ModelParameter():
    def __init__(self, **model_params) -> None:
        
        """
        dataDirectory
        modelFile
        initializingNumber
        lossFunction
        learningGoal
        learningRate
        learningRateLowerBound
        optimizer
        tuningTimes
        regularizingStrength
        """
        self.model_params = model_params
        
        self.dataDirectory = self.model_params["dataDirectory"]
        self.modelFile = self.model_params["modelFile"]
        self.initializingNumber = self.model_params["initializingNumber"]
        self.lossFunction = self.model_params["lossFunction"]
        self.learningGoal = self.model_params["learningGoal"]
        self.learningRate = self.model_params["learningRate"]
        self.learningRateLowerBound = self.model_params["learningRateLowerBound"]
        self.optimizer = self.model_params["optimizer"]
        self.tuningTimes = self.model_params["tuningTimes"]
        self.regularizingStrength = self.model_params["regularizingStrength"]

    def info(self):
        
        print(f'self.dataDirectory = {self.dataDirectory}')
        print(f'self.modelFile = {self.modelFile}')
        print(f'self.initializingNumber = {self.initializingNumber}')
        print(f'self.lossFunction = {self.lossFunction}')
        print(f'self.learningGoal = {self.learningGoal}')
        print(f'self.learningRate = {self.learningRate}')
        print(f'self.learningRateLowerBound = {self.learningRateLowerBound}')
        print(f'self.optimizer = {self.optimizer}')
        print(f'self.tuningTimes = {self.tuningTimes}')
        print(f'self.regularizingStrength = {self.regularizingStrength}')

class ModelParameter2(ModelParameter):
    def __init__(self, **model_params):

        """
        dataDirectory
        dataDescribing
        neuroNode
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

        self.model_params = model_params

    def info(self):
        
        print(f'self.model_params = {self.model_params}')
