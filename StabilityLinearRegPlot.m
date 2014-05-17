function [] = StabilityLinearRegPlot(file)
M = load(file)
ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
lr_results=[]
dropout_results=[]
detdropout_results=[]
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    lr_results(index)=mean(f('LinearReg'))
    dropout_results(index)=mean(f('DropoutLinearReg'))
end
plot(ratios(1:5), abs(diff(lr_results)), 'o-', ratios(1:5), abs(diff(dropout_results)), 'o-')
legend('LR', 'Dropout')
title('Difference in mean test errors between consecutive proportions of training data')
xlabel('Proportion')
ylabel('Difference in mean test error')
end

