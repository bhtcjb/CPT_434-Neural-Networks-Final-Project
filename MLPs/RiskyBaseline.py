import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from scipy.special import logit, expit

# this wrapper makes fractional logistic regression, since scikit's logistic regression is not continuous
class BoundedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
    
    # accepts groups so that Group K-fold does not cause errors
    def fit(self, x, y, groups=None):
        # y must be clipped to avoid log(0) or log(1)
        yLogit = logit(np.clip(y, 1e-6, 1 - 1e-6))
        self.model.fit(x, yLogit)
        self.fitted_ = True
        return self
    
    def predict(self, x):
        yLogit = self.model.predict(x)
        y = expit(yLogit)
        return y

# hyperparameters
numFeatures = 8
numClasses = 1

# grid search parameters
searchParameters = {
    'model__fit_intercept': [True, False],
}

def runValidation(x, y, groups):
    
    # set model defaults
    risqueModel = BoundedLinearRegression()
    
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
            n_jobs=1,
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
    Mean accuracy: -1.2365327715873717
    StdDev accuracy: 1.4454051970820916
    Median accuracy: -0.8945703506469727
    Best accuracy: 0.19400697946548462
    Best features: Index(['Fp1_Beta_Mean', 'TP10_Delta_Mean', 'TP10_Gamma_Mean', 'left outer HbO',
       'right outer HbO', 'left outer HbR', 'right outer HbR',
       'right inner HbO'],
      dtype='str')
    Best Parameters: {'model__fit_intercept': True}
"""
    