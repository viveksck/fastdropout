addpath(genpath('binaryLRloss'));
load('windowsgold.mat');
% Make sure we allways have the same split of training and test data
rng(0, 'twister')
%rand('seed', 0)
C = cvpartition(y, 'kfold',2);
Xtest = X(C.test(1),:);
ytest = y(C.test(1));
Xtrain = X(C.test(2),:);
ytrain = y(C.test(2));
output= run_stability_logistic_regression_experiment(Xtrain, ytrain, Xtest, ytest, [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 20, 0.0);
save('LogisticRegressionStabilityWindowsFine', 'output')
