function [] = StabilityLRPlot(file)
M = load(file)
ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
lr_results=[]
dropout_results=[]
detdropout_results=[]
for index=1:length(ratios)
    ratio=ratios(index)
    f = M.output(ratio)
    lr_results(index)=mean(f('LR'))
    dropout_results(index)=mean(f('Dropout'))
    detdropout_results(index)=mean(f('DetDropout'))
end
plot(ratios(1:5), abs(diff(lr_results)), ratios(1:5), abs(diff(dropout_results)), ratios(1:5), abs(diff(detdropout_results)))
legend('LR', 'Dropout', 'DetDropout')

