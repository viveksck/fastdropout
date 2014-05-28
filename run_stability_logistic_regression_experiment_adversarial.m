function [output] = run_stability_logistic_regression_experiment_adversarial(Xtrain, ytrain, Xtest, ytest, ratios, numRuns, model_file)
output=containers.Map('KeyType','double','ValueType','any');
for i=1:length(ratios)
    ratio = ratios(i);
    output(ratio)=containers.Map();
    temp=output(ratio);
    temp('LR')=zeros(numRuns,1);
    temp('Dropout')=zeros(numRuns,1);
    temp('DetDropout')=zeros(numRuns,1);
    for j=1:numRuns
        temp_lr = temp('LR');
        temp_dropout = temp('Dropout');
        temp_det_dropout = temp('DetDropout');
        
        results = stability_logistic_regression_adversarial(Xtrain, ytrain, Xtest, ytest,ratio, model_file);
        temp_lr(j) = results('LR');
        temp_dropout(j) = results('Dropout');
        temp_det_dropout(j) = results('DetDropout');
        
        temp('LR')=temp_lr;
        temp('Dropout')=temp_dropout;
        temp('DetDropout')=temp_det_dropout;
    end
end
end
