import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression

# we need a wrapper for logistic regression that accepts groups so that Group K-fold doesn't cause errors
class LogisticRegressionWrapper(LogisticRegression):
    def fit(self, x, y, groups=None):
        return super().fit(x, y)

# hyperparameters
features = ['Fp1_Beta_Mean', 'Fp1_Gamma_Mean', 'Fp2_Beta_Mean', 'Fp2_Gamma_Mean',
       'TP10_Delta_Mean', 'TP10_Theta_Mean', 'TP10_Beta_Mean', 'TP10_Gamma_Mean']
        
inputSize = len(features)
numClasses = 1

# grid search parameters
searchParameters = {
    'C': [1.0, 10, 100],
}


def runValidation(x, y, groups):
    
    # set model defaults
    hotModel = LogisticRegressionWrapper()

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
            n_jobs=1,
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
    Mean accuracy: 0.5872628458498024
    StdDev accuracy: 0.14020253631158375
    Median accuracy: 0.6594202898550725
    Best accuracy: 0.7272727272727273
    Best Parameters: {'C': 1.0}
"""
    