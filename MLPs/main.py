import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import MSELoss, BCELoss
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.callbacks import EpochScoring
from RiskyPerceptron import RiskyPerceptron
from HotPerceptron import HotPerceptron


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# selected best features
features = ['Fp1_Beta_Mean', 'TP10_Delta_Mean', 'TP10_Gamma_Mean', 'left outer HbO',
       'right outer HbO', 'left outer HbR', 'right outer HbR',
       'right inner HbO']

# selected best hyperparameters
risqueHyperparameters = {
    'batch_size': 4, 
    'lr': 1, 
    'max_epochs': 15, 
    'hiddenSize': 4, 
    'weight_decay': 0.0001
}

hotHyperparameters = {
    'batch_size': 16, 
    'lr': 0.1, 
    'max_epochs': 35, 
    'hiddenSize': 16, 
    'weight_decay': 0.001
}

# other hyperparameters
inputSize = len(features)
numClasses = 1
nesterov = True
momentum = 0.9


def trainRiskyModel(x, y):
    
    train_acc_callback = EpochScoring(
        scoring='r2',
        lower_is_better=False,
        name='train_acc',
        on_train=True
    )
    
    risqueModel = NeuralNetRegressor(
        module=RiskyPerceptron,
        module__inputSize=inputSize,
        module__hiddenSize=risqueHyperparameters['hiddenSize'],
        module__numClasses=numClasses,
        criterion=MSELoss,
        optimizer=torch.optim.SGD,
        lr=risqueHyperparameters['lr'],
        optimizer__momentum=momentum,
        optimizer__nesterov=nesterov,
        optimizer__weight_decay=risqueHyperparameters['weight_decay'],
        max_epochs=risqueHyperparameters['max_epochs'],
        batch_size=risqueHyperparameters['batch_size'],
        device=device,
        train_split=None,
        callbacks=[train_acc_callback],
        verbose=1,
    )
    
    risqueModel.fit(x, y)
    return risqueModel


def trainHotModel(x, y):
    
    train_acc_callback = EpochScoring(
        scoring='accuracy',
        lower_is_better=False,
        name='train_acc',
        on_train=True
    )
    
    hotModel = NeuralNetClassifier(
        module=HotPerceptron,
        module__inputSize=inputSize,
        module__hiddenSize=hotHyperparameters['hiddenSize'],
        module__numClasses=numClasses,
        criterion=BCELoss,
        optimizer=torch.optim.SGD,
        lr=hotHyperparameters['lr'],
        optimizer__momentum=momentum,
        optimizer__nesterov=nesterov,
        optimizer__weight_decay=hotHyperparameters['weight_decay'],
        max_epochs=hotHyperparameters['max_epochs'],
        batch_size=hotHyperparameters['batch_size'],
        device=device,
        train_split=None,
        callbacks=[train_acc_callback],
        verbose=1,
    )
    
    hotModel.fit(x, y)
    return hotModel


def graphTraining(risqueModel, hotModel):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # extract history for risque model
    risqueTrainLosses = []
    risqueTrainAccuracies = []
    if hasattr(risqueModel, 'history_'):
        for epoch in risqueModel.history_:
            if 'train_loss' in epoch:
                risqueTrainLosses.append(epoch['train_loss'])
            if 'train_acc' in epoch:
                risqueTrainAccuracies.append(epoch['train_acc'])
    
    # extract history for hot model
    hotTrainLosses = []
    hotTrainAccuracies = []
    if hasattr(hotModel, 'history_'):
        for epoch in hotModel.history_:
            if 'train_loss' in epoch:
                hotTrainLosses.append(epoch['train_loss'])
            if 'train_acc' in epoch:
                hotTrainAccuracies.append(epoch['train_acc'])
    
    risqueEpochs = range(1, len(risqueTrainLosses) + 1)
    hotEpochs = range(1, len(hotTrainLosses) + 1)
    
    # plot risque model loss
    axes[0, 0].plot(risqueEpochs, risqueTrainLosses, marker='o', color='steelblue')
    axes[0, 0].set_title('BART Performance Model - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].grid(True)
    
    # plot risque model accuracy
    axes[0, 1].plot(risqueEpochs, risqueTrainAccuracies, marker='o', color='steelblue')
    axes[0, 1].set_title('BART Performance Model - Accuracy')
    axes[0, 1].set_ylabel('R² Score')    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True)
    
    # plot hot model loss
    axes[1, 0].plot(hotEpochs, hotTrainLosses, marker='s', color='orange')
    axes[1, 0].set_title('Temperture Condition Model - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BCE Loss')
    axes[1, 0].grid(True)
    
    # plot hot model accuracy
    axes[1, 1].plot(hotEpochs, hotTrainAccuracies, marker='s', color='orange')
    axes[1, 1].set_title('Temperture Condition Model - Accuracy')    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Classification Accuracy')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_graphs.png', dpi=300)
    plt.show()


def calculateOutputCorrelation(risqueModel, hotModel, x):

    risquePrediction = risqueModel.predict(x)
    hotPrediction = hotModel.predict(x)
    
    correlation = np.corrcoef(risquePrediction, hotPrediction)[0, 1]
    
    return correlation


def calculateAverageGradientSaliency(model, x):

    xGrad = torch.from_numpy(x).float().to(device)
    xGrad.requires_grad_(True)
    
    # run forward pass
    output = model.module_(xGrad)
    
    # get gradients
    loss = output.sum()
    loss.backward()
    
    # get saliencies
    saliencies = torch.abs(xGrad.grad).detach().cpu().numpy()
    
    # calculate average saliency per feature
    avgSaliencies = np.mean(saliencies, axis=0)
    
    return avgSaliencies, saliencies


def plotGradientSaliency(risqueAvgSaliency, hotAvgSaliency):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # risque model saliencies
    axes[0].bar(range(len(risqueAvgSaliency)), risqueAvgSaliency, color='steelblue')
    axes[0].set_xticks(range(len(features)))
    axes[0].set_xticklabels(features, rotation=45, ha='right')
    axes[0].set_title('BART Performance Model - Average Gradient Saliency')
    axes[0].set_ylabel('Saliency')
    axes[0].grid(True, alpha=0.3)
    
    # hot model saliencies
    axes[1].bar(range(len(hotAvgSaliency)), hotAvgSaliency, color='orange')
    axes[1].set_xticks(range(len(features)))
    axes[1].set_xticklabels(features, rotation=45, ha='right')
    axes[1].set_title('Temperture Condition Model - Average Gradient Saliency')
    axes[1].set_ylabel('Saliency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_saliency_graphs.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    
    # read in data
    dataFrame = pd.read_csv('input/Training_Data.csv')
    
    # get BART samples
    bartData = dataFrame.dropna(subset=['Risk', 'Condition'])
    
    # risque model is trained on BART samples
    xRisque = bartData.filter(features).values.astype(np.float32)
    yRisque = bartData['Risk'].values.astype(np.float32).ravel()
    
    # isolate BART condition encodings for model comparisons
    yHotBartOnly = bartData['Condition'].values.astype(np.float32).ravel()
    
    # hot model is trained on all samples
    hotData = dataFrame.dropna(subset=['Condition'])
    xHot = hotData.filter(features).values.astype(np.float32)
    yHot = hotData['Condition'].values.astype(np.float32).ravel()
    
    # train models
    risqueModel = trainRiskyModel(xRisque, yRisque)
    hotModel = trainHotModel(xHot, yHot)
    
    # calculate model accuracies
    risqueScore = risqueModel.score(xRisque, yRisque)
    hotScore = hotModel.score(xHot, yHot)
    print(f"Risque Model Final Score: {risqueScore:.4f}")
    print(f"Hot Model Final Score: {hotScore:.4f}")
    
    # graph trainings
    graphTraining(risqueModel, hotModel)
    
    # calculate output correlation
    correlation = calculateOutputCorrelation(risqueModel, hotModel, xRisque)
    print(f"\nOutput Correlation: {correlation:.4f}")
    
    # calculate gradient saliencies
    risqueAvgSaliencies, risqueSaliencies = calculateAverageGradientSaliency(
        risqueModel, xRisque
    )
    hotAvgSaliencies, hotSaliencies = calculateAverageGradientSaliency(
        hotModel, xRisque
    )
    print("\nRisque Model Average Gradient Saliencies per Feature:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {risqueAvgSaliencies[i]:.6f}")
    
    print("\nHot Model Average Gradient Saliencies per Feature:")
    for i, feature in enumerate(features):
        print(f"  {feature}: {hotAvgSaliencies[i]:.6f}")
    print('\n')
    
    # plot saliencies
    plotGradientSaliency(risqueAvgSaliencies, hotAvgSaliencies)