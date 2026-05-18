import torch.nn as nn

class RiskyPerceptron(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(RiskyPerceptron, self).__init__()
        self.inputLayer = nn.Linear(inputSize, hiddenSize)
        self.outputLayer = nn.Linear(hiddenSize, numClasses)
        self.sigmoidActivation = nn.Sigmoid()

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.sigmoidActivation(out)
        out = self.outputLayer(out)
        out = self.sigmoidActivation(out)
        
        return out.view(-1)
    


