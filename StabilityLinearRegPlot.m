function [] = StabilityLinearRegPlot(file)
M = load(file)
ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
lr_results=[]
dropout_results=[]
detdropout_results=[]
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    lr_results(index)=mean(f('LinearReg'))
    dropout_results(index)=mean(f('DropoutLinearReg'))
end
hold on
plot(ratios, abs(lr_results - lr_results(1)), 'bo-', 'LineWidth', 2)
plot(ratios, abs(dropout_results - dropout_results(1)), 'go-', 'LineWidth', 2)
legend('LinearReg', 'Dropout')
xlabel('Proportion of training data removed')
ylabel('Difference in mean test error')
end
