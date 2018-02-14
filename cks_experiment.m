addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/SSDGP/GPstuff-4.6'));

solar = 0;
concrete = 0;
mauna = 0;
pp=0;
gefcom_rand_sub=1;
tidal_rand_sub=0;

siris = 0;
anzu = 1;
greypartridge = 0;
greyplover = 0;
greywagtail = 0;
greyheron = 0;
greyostrich = 0;

if solar
    load solar.mat
    x=X; Y=y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
    data_str = 'solar';
end
if concrete
    load concrete.mat
    x=X; Y=y;
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std;
    data_str = 'concrete';
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
    data_str = 'mauna';
end
if pp
    load pp.mat
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'pp';
end
if gefcom_rand_sub
    load gefcom.mat
    [n,D] = size(times);
    x = times; y=loads(:,1);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'gefcom';
end
if tidal_rand_sub
    load tidal.mat
    n = length(time_converted);
    x = time_converted; y = elevation;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'tidal_rand_sub';
end

if siris
    dir_str = '/data/siris/not-backed-up/hkim/';
end
if anzu
    dir_str = '/data/anzu/not-backed-up/hkim/';
end
if greypartridge
    dir_str = '/data/greypartridge/not-backed-up/oxwasp/oxwaspor/hkim/';
end
if greyplover
    dir_str = '/data/greyplover/not-backed-up/oxwasp/oxwaspor/hkim/';
end
if greywagtail
    dir_str = '/data/greywagtail/not-backed-up/oxwasp/oxwaspor/hkim/';
end
if greyheron
    dir_str = '/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/';
end
if greyostrich
    dir_str = '/data/greyostrich/not-backed-up/oxwasp/oxwaspor/hkim/';
end

if siris || anzu || greypartridge || greyplover || greyostrich
    myCluster = parcluster('local');
    myCluster.NumWorkers = 30;
    saveProfile(myCluster);
end

num_workers=15;
POOL=parpool('local',num_workers,'IdleTimeout',Inf);
final_depth=6;
num_iter=5;
seed=123;
S=1;

for subset_size = [320,640,1280,2560]
    if tidal_rand_sub || gefcom_rand_sub
        m = subset_size;
        string_txt = [dir_str data_str '_cks_experiment_' num2str(m) 'sub_' num2str(num_iter) 'iter.txt'];
        string = [dir_str data_str '_cks_experiment_' num2str(m) 'sub_' num2str(num_iter) 'iter.mat'];
    else
        m = n;
        string_txt = [dir_str data_str '_cks_experiment_' num2str(num_iter) 'iter.txt'];
        string = [dir_str data_str '_cks_experiment_' num2str(num_iter) 'iter.mat'];
    end
    ind = randsample(n,m);
    x_sub = x(ind);
    y_sub = y(ind);
    
    diary(string_txt);
    
    [kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = cks_parallel(x_sub,y_sub,final_depth,num_iter,seed,S,string);
    diary off
end


%save /data/greypartridge/not-backed-up/oxwasp/oxwaspor/hkim/concrete_kernel_tree_search_new.mat kernel_dict kernel_dict_debug
delete(POOL)
