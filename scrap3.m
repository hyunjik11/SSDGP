figure();
hold on
m_values=[1,2,3,4,5,6];
%nll = -2029.864; % mauna full GP
%nll = 83.864; % solar full GP
%nll = 248.82; % concrete full GP
%ne = -89.3388; % solar full GP
%ne = 2002.6; % mauna full GP
ne = -294.1799; % concrete full GP
plot(m_values,ne*ones(size(m_values)),'b');
scatter(reshape(repmat(m_values,10,1),60,1),reshape(lb_table,60,1),'gx');
plot(m_values,max(lb_table),'g');
scatter(reshape(repmat(m_values,10,1),60,1),reshape(approx_ub_table,60,1),'rx');
plot(m_values,max(approx_ub_table),'r');
%plot(m_values,max(ub_fic_table));
%plot(m_values,max(ub_pic_table));
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('fullGP','VAR LB','highest LB','UB', 'highest UB')
ylabel('negative energy')
xlabel('m')
hold off
saveas(gcf,'plots/approx_ub_concrete.fig')

for idx=1:length(kernel_dict)
    keys = kernel_dict.keys; key = keys{idx};
    val = kernel_dict(key);
    lb_vector = val{1};
    gp_ne = val{7};
    fprintf([key ' has gp_ne = %4.3f, lb = %4.3f \n'],gp_ne,lb_vector(6));
end