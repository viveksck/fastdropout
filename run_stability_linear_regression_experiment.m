function [output] = run_stability_linear_regression_experiment(Xtrain, ytrain, Xtest, ytest, ratios, numRuns, l2reg)
output=containers.Map('KeyType','double','ValueType','any');
for i=1:length(ratios)
    ratio = ratios(i);
    output(ratio)=containers.Map();
    temp=output(ratio);
    temp('LinearReg')=zeros(numRuns,1);
    temp('DropoutLinearReg')=zeros(numRuns,1);
    for j=1:numRuns
        temp_lr = temp('LinearReg');
        temp_dropout = temp('DropoutLinearReg');
        results = stability_linear_regression(Xtrain, ytrain, Xtest, ytest, ratio, l2reg);
        temp_lr(j) = results('LinearReg');
        temp_dropout(j) = results('DropoutLinearReg');
        temp('LinearReg')=temp_lr;
        temp('DropoutLinearReg')=temp_dropout;
    end
end
end