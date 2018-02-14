addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/SSDGP/GPstuff-4.6'));

solar = 0;
concrete = 0;
mauna = 0;
pp=0;
gefcom=1;
temp_ca_sb=0;
temp_ca_sb_ext=0;
temp_ca_sb_short=0;
tidal = 0;

siris = 0;
anzu = 0;
greypartridge = 0;
greyplover = 0;
greywagtail = 1;
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
if gefcom
    load gefcom.mat
    [n,D] = size(times);
    x = times; y=loads(:,1);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'gefcom';
end
if temp_ca_sb
    load temp_ca_sb.mat
    n = length(time_converted);
    x = time_converted; y = temp_avg;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'temp_ca_sb';
end
if temp_ca_sb_ext
    load temp_ca_sb_ext.mat
    n = length(time_converted);
    x = time_converted; y = temp_avg;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'temp_ca_sb_ext';
end
if temp_ca_sb_short
    load temp_ca_sb_short.mat
    n = length(time_converted);
    x = time_converted; y = temp_avg;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'temp_ca_sb_ext';
end
if tidal
    load tidal.mat
    n = length(time_converted);
    x = time_converted; y = elevation;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    data_str = 'tidal';
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
lb=0;
S=1;
seed=123;
precond = 'PIC';
%m_values = [10,20,40,80,160,320];

for m=[320]
        %if (m > 80) || ((m==80) && (S == 3))
        
        if lb==1
            string_txt = [dir_str data_str '_skc_lb_experiment_' num2str(m) 'm_' num2str(S), 'S' num2str(num_iter) 'iter.txt'];
            diary(string_txt);
            fprintf('m=%d, S=%d \n',m,S);
            string = [dir_str data_str '_skc_lb_experiment_' num2str(m) 'm_' num2str(S), 'S.mat'];
            [kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skc_parallel(x,y,final_depth,num_iter,m,seed,S,precond,string);
        else
            string_txt = [dir_str data_str '_skc_experiment_' num2str(m) 'm_' num2str(S), 'S' num2str(num_iter) 'iter.txt'];
            diary(string_txt);
            fprintf('m=%d, S=%d \n',m,S);
            string = [dir_str data_str '_skc_experiment_' num2str(m) 'm_' num2str(S), 'S.mat'];
            [kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skc_parallel_ub(x,y,final_depth,num_iter,m,seed,S,precond,string);
        end
        diary off
        %end
end


%m=20; S=3; final_depth=3;
%[kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skc_parallel_ub(x,y,final_depth,num_iter,m,seed,S,precond);
%plot_skc(kernel_buffer_history,kernel_top,m,S)
%[kernel_buffer, kernel_buffer_history, kernel_top, kernel_top_history] = skc(x,y,final_depth,num_iter,m_values,seed,S,precond);
%save /data/greypartridge/not-backed-up/oxwasp/oxwaspor/hkim/concrete_kernel_tree_search_new.mat kernel_dict kernel_dict_debug
%kernel_tree_plot(kernel_dict,m_values,[-1500,-1200],'plots/concrete_tree/');

delete(POOL)
