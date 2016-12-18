m=80; S=2;
figure()
hold on
ind = 1;
load('/data/siris/not-backed-up/hkim/solar_cks_experiment.mat');
subplot(3,2,1);
plot_cks(kernel_buffer_history,kernel_top)
title('CKS Solar') %top=((SE)*PER)*PER
xlim([-400,-100])
set(gca,'fontsize',14)
load('/data/siris/not-backed-up/hkim/solar_skd_experiment_80m_2S.mat');
subplot(3,2,2);
plot_skd(kernel_buffer_history,kernel_top,m,S)
title('SKC Solar, m=80, S=2') %top=(SE)*PER
xlim([-400,-100])
set(gca,'fontsize',14)

load('/data/siris/not-backed-up/hkim/mauna_cks_experiment.mat');
subplot(3,2,3);
plot_cks(kernel_buffer_history,kernel_top)
title('CKS Mauna') %top=(((((SE)+PER)*SE)+LIN)+SE)+SE
%xlim([600,2000])
set(gca,'fontsize',14)
load('/data/siris/not-backed-up/hkim/mauna_skd_experiment_80m_2S.mat');
subplot(3,2,4);
plot_skd(kernel_buffer_history,kernel_top,m,S)
title('SKC Mauna, m=80, S=2') %top=(((SE)+PER)*SE)+LIN
%xlim([600,2000])
set(gca,'fontsize',14)

S=1;
load('/data/siris/not-backed-up/hkim/concrete_cks_experiment.mat');
subplot(3,2,5);
plot_cks(kernel_buffer_history,kernel_top)
title('CKS Concrete') %top=(((((SE8)+SE1)*PER7)*SE2)+SE4)+LIN1
xlim([-1300,-350])
set(gca,'fontsize',14)
load('/data/siris/not-backed-up/hkim/concrete_skd_experiment_80m_1S.mat');
subplot(3,2,6);
plot_skd(kernel_buffer_history,kernel_top,m,S)
title('SKC Concrete, m=80, S=1') %top=(((((SE8)+SE1)+SE2)+SE4)+SE3)+SE7
xlim([-1300,-350])
set(gca,'fontsize',14)

hold off