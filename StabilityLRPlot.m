function [] = StabilityLRPlot(file)
M = load(file);
ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'
%ratios=[0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]';
lr_results=[];
dropout_results=[];
detdropout_results=[];
for index=1:length(ratios)
    ratio=ratios(index);
    f = M.output(ratio);
    lr_results(index)=mean(f('LR'));
    dropout_results(index)=mean(f('Dropout'));
    detdropout_results(index)=mean(f('DetDropout'));
end
figure(1)
hold on
plot(ratios, abs(lr_results - lr_results(1)), 'bo-', 'LineWidth', 2)
plot(ratios, abs(dropout_results - dropout_results(1)), 'go-', 'LineWidth', 2)
plot(ratios, abs(detdropout_results - detdropout_results(1)), 'ro-', 'LineWidth', 2)
outputfilename=sprintf('python_%s',file);
save(outputfilename, 'ratios', 'lr_results', 'dropout_results', 'detdropout_results')
legend('LR', 'Dropout', 'DetDropout')
xlabel('Proportion of training data removed')
ylabel('Difference in mean test error')
