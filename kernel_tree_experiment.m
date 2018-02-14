addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'))
solar = 0;
concrete = 1;
mauna = 0;

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

num_workers=10;
POOL=parpool('local',num_workers);

final_depth=1;
num_iter=10;
seed=123;
fullgp=1;
m_values = [10,20,40,80,160,320];

[kernel_dict, kernel_dict_debug] = kernel_tree(x,y,final_depth,num_iter,m_values,seed,fullgp,'PIC');
%save /data/anzu/not-backed-up/hkim/concrete_kernel_tree_search_new.mat kernel_dict kernel_dict_debug
kernel_tree_plot(kernel_dict,m_values,[-1500,-1200],'plots/concrete_tree/');
kernel_tree_plot(kernel_dict,m_values,[0,2000],'plots/mauna_tree/');
kernel_tree_plot(kernel_dict,m_values,[-500,-100],'plots/solar_tree/');
delete(POOL)