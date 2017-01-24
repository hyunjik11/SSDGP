figure()
hold on
ind = 1;
S_values = [2]; m_values = [80,160,320];
for m = m_values
    for S = S_values
        % /not-backed-up/oxwasp/oxwaspor
        string = ['/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/pp_skc_experiment_' num2str(m) 'm_' num2str(S), 'S.mat'];
        load(string);
        
        subplot(length(S_values),length(m_values),ind);
        plot_skc(kernel_buffer_history,kernel_top,m,S)
        xlim([-2000,2000])
        
        ind = ind + 1;
    end
end
hold off