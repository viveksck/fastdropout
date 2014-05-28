function [] = StabilityLRPlotTestError(file)
M = load(file);
ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'
lr_results=[];
dropout_results=[];
detdropout_results=[];
lr_results_error=[];
dropout_results_error=[];
detdropout_results_error=[];
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    %f('Key') stores the accuracy. so 1.0 - accuracy gives error.
    lr_results(index)=mean(1.0-f('LR'))
    lr_results_error(index)=std(1.0-f('LR'))/sqrt(length(f('LR')));
    dropout_results(index)=mean(1.0-f('Dropout'))
    dropout_results_error(index)=std(1.0-f('Dropout'))/sqrt(length(f('Dropout')));
    detdropout_results(index)=mean(1.0-f('DetDropout'))
    detdropout_results_error(index)=std(1.0-f('DetDropout'))/sqrt(length(f('DetDropout')));
end
figure(1)
hold on
errorbar(ratios, lr_results, lr_results_error, 'bo-', 'LineWidth', 2)
errorbar(ratios, dropout_results, dropout_results_error, 'go-', 'LineWidth', 2)
errorbar(ratios, detdropout_results, detdropout_results_error, 'ro-', 'LineWidth', 2)
legend('LR', 'Dropout', 'DetDropout')
%title('Difference in test error between full data.')
xlabel('Proportion of training data removed')
ylabel('Mean test error')