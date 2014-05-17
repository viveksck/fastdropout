function [results] = stability_linear_regression(Xtrain, ytrain, Xtest, ytest, ratio)
addpath(genpath('LinearRegLoss'));
% Make sure we always have the same split of training and test data
rng('shuffle')
% Hold out data only if required to.
if ratio > 0.0
  cv = cvpartition(ytrain, 'holdout', ratio);
  Xtrain = Xtrain(training(cv,1), :);
  ytrain = ytrain(training(cv,1), :);
  size(Xtrain)
end
%%
w_init = 0*randn(size(Xtrain,2),1);
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-5;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 10000;
mfOptions.DerivativeCheck = 0;
mfOptions.useMex=0;
results = containers.Map;
casenames = {'LinearReg', 'DropoutLinearReg'};
%casenames = {'LinearReg'};
for casenum = 1:length(casenames)
    obj = casenames{casenum};
    switch obj
        case 'LinearReg'
            funObj = @(w)LinearRegLoss(w,Xtrain,ytrain);
            lambdaL2 = 0.0;
        case 'DropoutLinearReg'
            funObj = @(w)LinRegLossMCDropoutSample(w,Xtrain,ytrain,1.0,100,100);
            lambdaL2 = 0.0;
    end
    
    funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
    w = minFunc(funObjL2,w_init,mfOptions);
    ypred = Xtest * w;
    acc = (norm(ypred-ytest)^2)/length(ypred);
    resultname = [casenames{casenum}];
    results(resultname) = acc;
end

keys = results.keys;
for i=1:length(keys)
    fprintf('%s: %f\n', keys{i}, results(keys{i}));
end
end


