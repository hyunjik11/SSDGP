addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
solar = 0;
concrete = 1;
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
S=1;

string_txt = ['/data/anzu/not-backed-up/hkim/concrete_cks_experiment.txt'];
diary(string_txt);
tic;
[kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = cks_parallel(x,y,final_depth,num_iter,seed,S);
toc;
diary off
string = ['/data/anzu/not-backed-up/hkim/concrete_cks_experiment.mat'];
save(string,'kernel_buffer','kernel_buffer_history', 'kernel_top', 'kernel_top_history');


%save /data/greypartridge/not-backed-up/oxwasp/oxwaspor/hkim/concrete_kernel_tree_search_new.mat kernel_dict kernel_dict_debug


delete(POOL)
