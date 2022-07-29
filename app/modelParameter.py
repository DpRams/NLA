class ModelParameter():
    def __init__(self, dataDirectory, modelFile, lossFunction, learningGoal, learningRate, learningRateUpperBound, learningRateLowerBound, optimizer, regularizingStrength) -> None:
        self.dataDirectory = dataDirectory
        self.modelFile = modelFile
        self.lossFunction = lossFunction
        self.learningGoal = learningGoal
        self.learningRate = learningRate
        self.learningRateUpperBound = learningRateUpperBound
        self.learningRateLowerBound = learningRateLowerBound
        self.optimizer = optimizer
        self.regularizingStrength = regularizingStrength
    
    def info(self):
        
        print(f'self.dataDirectory = {self.dataDirectory}')
        print(f'self.modelFile = {self.modelFile}')
        print(f'self.lossFunction = {self.lossFunction}')
        print(f'self.learningGoal = {self.learningGoal}')
        print(f'self.learningRate = {self.learningRate}')
        print(f'self.learningRateUpperBound = {self.learningRateUpperBound}')
        print(f'self.learningRateLowerBound = {self.learningRateLowerBound}')
        print(f'self.optimizer = {self.optimizer}')
        print(f'self.regularizingStrength = {self.regularizingStrength}')