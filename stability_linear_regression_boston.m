M = load('housing.mat');
% Make sure we allways have the same split of training and test data
output= run_stability_linear_regression_experiment(M.Xtrain, M.ytrain, M.Xtest, M.ytest, [0.0, 0.05, 0.1, 0.15,  0.2, 0.3, 0.4, 0.5], 50);
save('LinearRegressionStabilityBostonFine', 'output')