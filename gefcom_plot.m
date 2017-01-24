% script to plot gefcom on different scales
load('gefcom.mat')
figure()
subplot(2,1,1);
plot(2004+(times-min(times))/365,loads(:,1))
ylabel('load (kW)'); xlabel('year');
subplot(2,1,2);
% plot on local scale
plot((times(1:24*14)-min(times)),loads(1:24*14,1))
ylabel('load (kW)'); xlabel('day in Jan 2004');