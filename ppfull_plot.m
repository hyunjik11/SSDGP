figure();
subplot(2,2,1);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_logdet*ones(size(m_values)));
errorbar(m_values,mean_ld_rff,std_ld_rff);
errorbar(m_values,mean_ld_naive,std_ld_naive);
errorbar(m_values,mean_ld_fic,std_ld_fic);
errorbar(m_values,mean_ld_pic,std_ld_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
ylim([-20000 -2000])
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('logdet/2')
xlabel('m')
hold off

subplot(2,2,2);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_innerprod*ones(size(m_values)));
errorbar(m_values,mean_ip_rff,std_ip_rff);
errorbar(m_values,mean_ip_naive,std_ip_naive);
errorbar(m_values,mean_ip_fic,std_ip_fic);
errorbar(m_values,mean_ip_pic,std_ip_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
ylim([3000 7000])
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('innerprod/2')
xlabel('m')
hold off

subplot(2,2,3);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,frob_svd);
errorbar(m_values,mean_frob_rff,std_frob_rff);
errorbar(m_values,mean_frob_naive,std_frob_naive);
errorbar(m_values,mean_frob_fic,std_frob_fic);
errorbar(m_values,mean_frob_pic,std_frob_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
ylim([0 900]);
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Frobenius Norm Error')
xlabel('m')
hold off

subplot(2,2,4);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,spec_svd);
errorbar(m_values,mean_spec_rff,std_spec_rff);
errorbar(m_values,mean_spec_naive,std_spec_naive);
errorbar(m_values,mean_spec_fic,std_spec_fic);
errorbar(m_values,mean_spec_pic,std_spec_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
ylim([0 250]);
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Spectral Norm Error')
xlabel('m')
hold off