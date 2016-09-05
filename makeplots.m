figure();
subplot(1,3,1);
hold on
m_values=[1,2,3,4,5,6];
nll = -2029.864; % mauna full GP
%nll = 83.864; % solar full GP
plot(m_values,-nll*ones(size(m_values)));
plot(m_values,ml);

scatter(reshape(repmat(m_values,10,1),60,1),reshape(lb_table,60,1),'gx');
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('fullGP','exact','Var LB')
ylabel('marginal log lkhd')
xlabel('m')
hold off

subplot(1,3,2);
hold on
plot(m_values,nld,'linewidth',2);
plot(m_values,naive_nld_table,'linewidth',1);
scatter(reshape(repmat(m_values,10,1),60,1),reshape(rff_nld_table,60,1),'x');
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('exact','Nystrom NLD UB','RFF NLD UB')
ylabel('NLD')
xlabel('m')
hold off

subplot(1,3,3);
hold on
plot(m_values,nip);
plot(m_values,cg_obj_table);
plot(m_values,pcg_obj_table,'linewidth',3);
plot(m_values,pcg_objf_table,'linewidth',2);
plot(m_values,pcg_objp_table,'linewidth',1);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('exact','CG','PCG Nys','PCG FIC','PCG PIC')
xlabel('m')
ylabel('NIP')
hold off
