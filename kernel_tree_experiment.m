addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
solar = 1;
concrete = 0;
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
%POOL=parpool('local',num_workers);

keySet = {'SE','LIN','PER'};
valueSet = {se_init(x,y),lin_init(),per_init(x,y)};
base_kernels=containers.Map(keySet,valueSet);

final_depth=2;
num_iter=10;
seed=123;
fullgp=1;
m_values = [10,20,40,80,160,320];

[kernel_dict, kernel_dict_debug] = kernel_tree(x,y,base_kernels,final_depth,num_iter,m_values,seed,fullgp);

kernel_tree_plot(kernel_dict,m_values);

delete(POOL)