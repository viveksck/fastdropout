M = load('housing.mat');
l2regs = [0.0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0 ,4.0, 8.0, 16.0]
for l2reg = l2regs
    output= run_stability_linear_regression_experiment(M.Xtrain, M.ytrain, M.Xtest, M.ytest, [0.0, 0.05, 0.1, 0.15,  0.2, 0.3, 0.4, 0.5], 50, l2reg);
    output_file = sprintf('%f_LinearRegressionStabilityBostonFine.mat', l2reg)
    save(output_file, 'output')
end