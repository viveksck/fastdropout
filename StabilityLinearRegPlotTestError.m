function [] = StabilityLinearRegPlotTestError(file)
M = load(file)
ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
lr_results=[]
dropout_results=[]
lr_results_error=[];
dropout_results_error=[];
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    lr_results(index)=mean(f('LinearReg'))
    lr_results_error(index)=std(f('LinearReg')/sqrt(length(f('LinearReg'))));
    dropout_results(index)=mean(f('DropoutLinearReg'))
    dropout_results_error(index)=std(f('DropoutLinearReg')/sqrt(length(f('DropoutLinearReg'))));
end
figure(1)
hold on
errorbar(ratios, lr_results, lr_results_error, 'bo-', 'LineWidth', 2)
errorbar(ratios, dropout_results, dropout_results_error, 'go-', 'LineWidth', 2)
legend('LinearReg', 'Dropout')
xlabel('Proportion of training data removed')
ylabel('Mean test error')
end
