# CPT_434-Neural-Networks-Final-Project
Pipeline code for final project for CPT_S 434 Neural Networks class


Read paper here: https://github.com/bhtcjb/CPT_434-Neural-Networks-Final-Project/blob/0130280d761ae0f53bbec22b32ba34d76e07546c/Identifying%20Temperature%20Condition%20and%20Risk-Taking%20Scores%20with%20Neural%20and%20Physiological%20Features.pdf

# Identifying Temperature Condition and Risk-Taking Scores with Neural and Physiological Features
Blake Turman
### Abstract— Non-invasive techniques to measure brain activity, such as electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS), can add a wall of depth to environmental and behavioral neuroscience research. However, analyzing complex multi-feature EEG and fNIRS data can introduce particular difficulty. This work proposes to analyze combined EEG and fNIRS data using artificial neural networks to determine the neural and neurophysiological features that link environmental temperature to risk-taking behavior. The method involves two shallow Multi-Layer Perceptron (MLP) models trained on EEG and fNIRS data. One model predicts hot vs neutral condition. The other model predicts risk-taking behavior. The results are compared to find the link between temperature condition and risk-taking behavior based on the input features.

<br>
Pipline: Clean and process EEG and fNIRS data (EEG and fNIRS process) + Construct groundtruth labels from BART data (Encode actuals) --> Put it all together and organize (Consolidate input data) --> Run neural network analyses (MLPs)
</br>
<br>
Analyses: Run BART perfomance model validation loop (RiskyValidation) --> Use selected best features to run temperature condition model validation loop (HotValidation) --> Run baseline validation loops likewise (RiskyBaseline --> HotBaseline) --> Run main with selected best features and hyperparameters
</br>
<br>
All real data is omitted from repository
