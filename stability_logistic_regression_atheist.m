addpath(genpath('binaryLRloss'));
load('example_data.mat');
% Make sure we allways have the same split of training and test data
rand('twister', 0)
C = cvpartition(y, 'kfold',2);
Xtest = X(C.test(1),:);
ytest = y(C.test(1));
Xtrain = X(C.test(2),:);
ytrain = y(C.test(2));
l2regs = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0 ,4.0, 8.0, 16.0]
for l2reg = l2regs
  l2reg
  output= run_stability_logistic_regression_experiment(Xtrain, ytrain, Xtest, ytest, [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 20, l2reg);
  output_file = sprintf('%f_LogisticRegressionStabilityAtheistFine.mat', l2reg)
  save(output_file, 'output')
end
