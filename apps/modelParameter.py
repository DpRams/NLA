class ModelParameter():
    def __init__(self, dataDirectory, modelFile, initializingNumber, lossFunction, learningGoal, learningRate, learningRateLowerBound, optimizer, tuningTimes, regularizingStrength) -> None:
        self.dataDirectory = dataDirectory
        self.modelFile = modelFile
        self.initializingNumber = initializingNumber
        self.lossFunction = lossFunction
        self.learningGoal = learningGoal
        self.learningRate = learningRate
        self.learningRateLowerBound = learningRateLowerBound
        self.optimizer = optimizer
        self.tuningTimes = tuningTimes
        self.regularizingStrength = regularizingStrength
    
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