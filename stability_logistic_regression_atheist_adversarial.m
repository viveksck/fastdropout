addpath(genpath('binaryLRloss'));
load('example_data.mat');
% Make sure we allways have the same split of training and test data
%rand('seed', 0)
rng(0, 'twister')
C = cvpartition(y, 'kfold',2);
Xtest = X(C.test(1),:);
ytest = y(C.test(1));
Xtrain = X(C.test(2),:);
ytrain = y(C.test(2));
ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
%ratios_fine = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
output=run_stability_logistic_regression_experiment_adversarial(Xtrain, ytrain, Xtest, ytest, ratios, 20, 'atheistLRFinemodel.mat',0);
save('LogisticRegressionStabilityAtheistAdversarialFine', 'output')
