function [] = StabilityLRPlot(file)
M = load(file);
ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'
%ratios= [0.0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2]
lr_results=[];
dropout_results=[];
detdropout_results=[];
lr_results_error=[];
dropout_results_error=[];
detdropout_results_error=[];
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    lr_results(index)=mean(f('LR'))
    lr_results_error(index)=std(f('LR')/sqrt(length(f('LR'))));
    dropout_results(index)=mean(f('Dropout'))
    dropout_results_error(index)=std(f('Dropout')/sqrt(length(f('Dropout'))));
    detdropout_results(index)=mean(f('DetDropout'))
    detdropout_results_error(index)=std(f('DetDropout')/sqrt(length(f('DetDropout'))));
end
figure(1)
hold on
plot(ratios, abs(lr_results - lr_results(1)), 'bo-', 'LineWidth', 2)
plot(ratios, abs(dropout_results - dropout_results(1)), 'go-', 'LineWidth', 2)
plot(ratios, abs(detdropout_results - detdropout_results(1)), 'ro-', 'LineWidth', 2)
%errorbar(ratios, abs(lr_results - lr_results(1)), lr_results_error, 'bo-', 'LineWidth', 2)
%errorbar(ratios, abs(dropout_results - dropout_results(1)), dropout_results_error, 'go-', 'LineWidth', 2)
%errorbar(ratios, abs(detdropout_results - detdropout_results(1)), detdropout_results_error, 'ro-', 'LineWidth', 2)
legend('LR', 'Dropout', 'DetDropout')
%title('Difference in test error between full data.')
xlabel('Proportion of training data removed')
ylabel('Difference in mean test error')