import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from skorch import NeuralNetRegressor
from torch.nn import MSELoss
from RiskyPerceptron import RiskyPerceptron


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
numFeatures = 8
numClasses = 1
nesterov = True
momentum = 0.9

# grid search parameters
searchParameters = {
    'model__lr': [1],
    'model__module__hiddenSize': [4],
    'model__max_epochs': [15],
    'model__batch_size': [2],
    'model__optimizer__weight_decay': [0.0001],
}

def runValidation(x, y, groups):
    
    # set model defaults
    risqueModel = NeuralNetRegressor(
        module=RiskyPerceptron,
        module__inputSize=numFeatures,
        module__hiddenSize=8,
        module__numClasses=numClasses,
        criterion=MSELoss,
        optimizer=torch.optim.SGD,
        lr=0.1,
        optimizer__momentum=momentum,
        optimizer__nesterov=nesterov,
        max_epochs=10,
        batch_size=16,
        device=device,
        verbose=0,
    )
    
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=mutual_info_regression, k=numFeatures)), 
        ('model', risqueModel)
    ])

    # create inner and outer cross validation folds
    innerFold = GroupKFold(n_splits=3)
    outerFold = GroupKFold(n_splits=5)

    # manually cross validate since cross_val_score() doesn't work for groups >:(
    accuraciesList = []
    bestParametersList = []
    bestFeaturesList = []
    for fold, (trainIndex, testIndex) in enumerate(outerFold.split(x, y, groups)):
        xTrain, yTrain = x[trainIndex], y[trainIndex]
        groupTrain = groups[trainIndex]
        xTest, yTest = x[testIndex], y[testIndex]

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=searchParameters,
            cv=innerFold,
            scoring='r2',
            refit=True,
            n_jobs=5,
            verbose=0,
        )

        grid.fit(xTrain, yTrain, groups=groupTrain)
        

        bestFeatures = grid.best_estimator_.named_steps['selector'].get_support()
        bestFeaturesList.append(bestFeatures) 
        bestParameters = grid.best_params_
        bestParametersList.append(bestParameters)
        accuracy = grid.score(xTest, yTest)
        accuraciesList.append(accuracy)
        
        print("Fold:", fold)

    
    return bestFeaturesList, bestParametersList, accuraciesList



if __name__ == '__main__':

    # read in data
    dataFrame = pd.read_csv('input/Training_Data.csv')

    # isolate BART data
    dataFrame = dataFrame.dropna(subset=['Risk'])

    # make input and output features
    x = dataFrame.drop(columns=['Sample', 'Condition', 'Risk']).values.astype(np.float32)
    y = dataFrame['Risk'].values.astype(np.float32).ravel() 

    # group by participant number
    samples = dataFrame['Sample'].astype(str)
    participantNumbers = samples.str.split('_').str[0]
    participants = pd.factorize(participantNumbers)

    bestFeatures, bestParameters, accuracies = runValidation(x, y, groups=participants[0])
    
    bestIndex = accuracies.index(max(accuracies))
    
    print('Mean accuracy:', np.mean(accuracies))
    print('StdDev accuracy:', np.std(accuracies))
    print('Median accuracy:', np.median(accuracies))
    print('Best accuracy:', accuracies[bestIndex])
    print('Best features:', dataFrame.drop(columns=['Sample', 'Condition', 'Risk']).columns[bestFeatures[bestIndex]])
    print('Best Parameters:', bestParameters[bestIndex])
    


"""

Mean accuracy: -2.375512790679932
StdDev accuracy: 4.195695027339628
Median accuracy: -0.292797327041626
Best accuracy: 0.14128601551055908
Best features: Index(['Fp1_Beta_Mean', 'TP10_Delta_Mean', 'TP10_Gamma_Mean', 'left outer HbO',
       'right outer HbO', 'left outer HbR', 'right outer HbR',
       'right inner HbO'],
      dtype='str')
Best Parameters: {'model__batch_size': 4, 'model__lr': 0.1, 'model__max_epochs': 10, 'model__module__hiddenSize': 4, 'model__optimizer__weight_decay': 0.01}

Mean accuracy: -1.3232925891876222
StdDev accuracy: 1.0037156924985033
Median accuracy: -1.575695276260376
Best accuracy: 0.238594651222229
Best features: Index(['Fp1_Beta_Mean', 'TP10_Delta_Mean', 'TP10_Gamma_Mean', 'left outer HbO',
       'right outer HbO', 'left outer HbR', 'right outer HbR',
       'right inner HbO'],
      dtype='str')
Best Parameters: {'model__batch_size': 4, 'model__lr': 1, 'model__max_epochs': 15, 'model__module__hiddenSize': 4, 'model__optimizer__weight_decay': 0.0001}

"""
