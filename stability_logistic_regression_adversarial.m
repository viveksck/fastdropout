function [results] = stability_logistic_regression_adversarial(Xtrain, ytrain, Xtest, ytest, ratio, model_file, num)
addpath(genpath('binaryLRloss'));
% Make sure we always have the same split of training and test data
% Hold out data only if required to.
if ratio > 0.0
  pw = load(model_file);
  proj = full(Xtrain) * pw.w;
  proj = abs(proj);
  Xtraintemp = [full(Xtrain) proj];
  ytraintemp = [ytrain proj];
  xsize = size(Xtraintemp);
  ysize = size(ytraintemp);
  Xtraintempsorted = sortrows(Xtraintemp, xsize(2));
  ytraintempsorted = sortrows(ytraintemp, ysize(2));
  
  Xtrainnew = Xtraintempsorted(:,1:xsize(2)-1);
  ytrainnew = ytraintempsorted(:,1:ysize(2)-1);
  if num == 1
    num_remove=ratio
  else 
    num_remove = ratio * xsize(1)
  end
  Xtrain = Xtrainnew((num_remove + 1):xsize(1),:);
  ytrain = ytrainnew((num_remove + 1):ysize(1),:); 
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
    if ratio == 0.0 
        if casenum == 1
            save(model_file, 'w')
        end
    end
end

keys = results.keys;
for i=1:length(keys)
    fprintf('%s: %f\n', keys{i}, results(keys{i}));
end
end
