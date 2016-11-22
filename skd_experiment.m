addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
solar = 1;
concrete = 0;
mauna = 0;
pp=0;

if solar
    load solar.mat
    x=X; Y=y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
end
if concrete
    load concrete.mat
    x=X; Y=y;
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std;
end

if mauna
    load mauna.txt
    z = mauna(:,2) ~= -99.99; % get rid of missing data
    x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
    X = x; Y = y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
end
if pp
    load pp.mat
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
end

num_workers=20;
POOL=parpool('local',num_workers);

final_depth=6;
num_iter=10;
seed=123;
precond = 'PIC';
%m_values = [10,20,40,80,160,320];
if 1==0
for m=[20,80,320]
    for S=1:3
        if (m==320) && (S>1)
        string_txt = ['/data/siris/not-backed-up/hkim/solar_skd_experiment_' num2str(m) 'm_' num2str(S), 'S.txt'];
        diary(string_txt);
        fprintf('m=%d, S=%d \n',m,S);
        tic;
        [kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skd_parallel(x,y,final_depth,num_iter,m,seed,S,precond);
        toc;
        diary off
        string = ['/data/siris/not-backed-up/hkim/solar_skd_experiment_' num2str(m) 'm_' num2str(S), 'S.mat'];
        save(string,'kernel_buffer', 'kernel_buffer_history', 'kernel_top', 'kernel_top_history');
        end
    end
end
end

m=20; S=2; final_depth=3;
[kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skd_parallel(x,y,final_depth,num_iter,m,seed,S,precond);
%plot_skd(kernel_buffer_history,m,S)
%[kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skd(x,y,final_depth,num_iter,m_values,seed,S,precond);
%save /data/greypartridge/not-backed-up/oxwasp/oxwaspor/hkim/concrete_kernel_tree_search_new.mat kernel_dict kernel_dict_debug
%kernel_tree_plot(kernel_dict,m_values,[-1500,-1200],'plots/concrete_tree/');

delete(POOL)
