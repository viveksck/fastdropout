function [results] = stability_logistic_regression(Xtrain, ytrain, Xtest, ytest, ratio)
addpath(genpath('binaryLRloss'));
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
mfOptions.optTol = 2e-2;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 1000;
mfOptions.DerivativeCheck = 0;
mfOptions.useMex=0;
results = containers.Map;
casenames = {'LR', 'DetDropout', 'Dropout'};
for casenum = 1:length(casenames)
    obj = casenames{casenum};
    switch obj
        case 'LR'
            funObj = @(w)LogisticLoss(w,Xtrain,ytrain);
            lambdaL2=0.00; 
% you can optimize this value on the test set,
% and LR would still be quite a bit worse
            
        case 'DetDropout'
            funObj = @(w)LogisticLossDetObjDropout(w,Xtrain,ytrain,0.5);
            lambdaL2=0.00;
            
        case 'DetDropoutApprox'
            funObj = @(w)LogisticLossDetObjDropoutDeltaApprox(w,Xtrain,ytrain,0.5);
            lambdaL2=0.00;
            
        case 'Dropout'
            funObj = @(w)LogisticLossMCDropoutSample(w,Xtrain,ytrain,0.5,100,100);
            lambdaL2=0.00;
    end
    
    funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
    w = minFunc(funObjL2,w_init,mfOptions);
    ypred = Xtest * w > 0;
    acc = sum(ypred == (ytest+1)/2 )/length(ytest);
    resultname = [casenames{casenum}];
    results(resultname) = acc;
end

keys = results.keys;
for i=1:length(keys)
    fprintf('%s: %f\n', keys{i}, results(keys{i}));
end
end