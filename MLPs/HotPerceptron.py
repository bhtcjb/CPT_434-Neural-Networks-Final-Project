import torch.nn as nn


class HotPerceptron(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(HotPerceptron, self).__init__()
        self.inputLayer = nn.Linear(inputSize, hiddenSize)
        self.hiddenLayer = nn.Linear(hiddenSize, hiddenSize)
        self.outputLayer = nn.Linear(hiddenSize, numClasses)
        self.reluActivation = nn.ReLU()
        self.sigmoidActivation = nn.Sigmoid()

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.reluActivation(out)
        out = self.hiddenLayer(out)
        out = self.reluActivation(out)
        out = self.outputLayer(out)
        out = self.sigmoidActivation(out)
        
        return out.view(-1)
