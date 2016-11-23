figure()
hold on
ind = 1;
S_values = [1,2,3]; m_values = [20,80,320];
for m = m_values
    for S = S_values
        string = ['/data/anzu/not-backed-up/hkim/mauna_skd_experiment_' num2str(m) 'm_' num2str(S), 'S.mat'];
        load(string);
        
        subplot(length(S_values),length(m_values),ind);
        plot_skd(kernel_buffer_history,kernel_top,m,S)
        xlim([600,2000])
        
        ind = ind + 1;
    end
end
hold off