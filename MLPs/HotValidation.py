import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, GroupKFold
from skorch import NeuralNetClassifier
from torch.nn import BCELoss
from HotPerceptron import HotPerceptron

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
features = ['Fp1_Beta_Mean', 'TP10_Delta_Mean', 'TP10_Gamma_Mean', 'left outer HbO',
       'right outer HbO', 'left outer HbR', 'right outer HbR',
       'right inner HbO']
        
inputSize = len(features)
numClasses = 1
nesterov = True
momentum = 0.9

# grid search parameters
searchParameters = {
    'lr': [0.1],
    'module__hiddenSize': [16],
    'max_epochs': [35],
    'batch_size': [16],
    'optimizer__weight_decay': [0.001],
}


def runValidation(x, y, groups):
    
    # set model defaults
    hotModel = NeuralNetClassifier(
        module=HotPerceptron,
        module__inputSize=inputSize,
        module__hiddenSize=8,
        module__numClasses=numClasses,
        criterion=BCELoss,
        optimizer=torch.optim.SGD,
        lr=0.1,
        optimizer__momentum=momentum,
        optimizer__nesterov=nesterov,
        max_epochs=10,
        batch_size=16,
        device=device,
        verbose=0,
    )

    # create inner and outer cross validation folds
    innerFold = GroupKFold(n_splits=5)
    outerFold = GroupKFold(n_splits=10)

    # manually cross validate since cross_val_score() doesn't work for groups >:(
    accuraciesList = []
    bestParametersList = []
    for fold, (trainIndex, testIndex) in enumerate(outerFold.split(x, y, groups)):
        xTrain, yTrain = x[trainIndex], y[trainIndex]
        groupTrain = groups[trainIndex]
        xTest, yTest = x[testIndex], y[testIndex]

        grid = GridSearchCV(
            estimator=hotModel,
            param_grid=searchParameters,
            cv=innerFold,
            scoring='accuracy',
            refit=True,
            n_jobs=4,
            verbose=0,
        )

        grid.fit(xTrain, yTrain, groups=groupTrain)
        bestParameters = grid.best_params_
        bestParametersList.append(bestParameters)
        accuracy = grid.score(xTest, yTest)
        accuraciesList.append(accuracy)

        print("Fold:", fold)
    
    return bestParametersList, accuraciesList


if __name__ == '__main__':

    # read in data
    dataFrame = pd.read_csv('input/Training_Data.csv')
    
    # select features
    dataFrame = dataFrame.filter(['Sample', 'Condition', 'Risk'] + features, axis=1)

    # make input and output features
    x = dataFrame.drop(columns=['Sample', 'Condition', 'Risk']).values.astype(np.float32)
    y = dataFrame['Condition'].values.astype(np.float32).ravel() 

    # group by participant number
    samples = dataFrame['Sample'].astype(str)
    participantNumbers = samples.str.split('_').str[0]
    participants = pd.factorize(participantNumbers)

    bestParameters, accuracies = runValidation(x, y, groups=participants[0])
    
    bestIndex = accuracies.index(max(accuracies))
    
    print('Mean accuracy:', np.mean(accuracies))
    print('StdDev accuracy:', np.std(accuracies))
    print('Median accuracy:', np.median(accuracies))
    print('Best accuracy:', accuracies[bestIndex])
    print('Best Parameters:', bestParameters[bestIndex])
    

"""

Mean accuracy: 0.5375098814229249
StdDev accuracy: 0.20380465846850868
Median accuracy: 0.5871212121212122
Best accuracy: 0.9166666666666666
Best Parameters: {'batch_size': 16, 'lr': 0.1, 'max_epochs': 35, 'module__hiddenSize': 16, 'optimizer__weight_decay': 0.001}

"""
