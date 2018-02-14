% subplot(1,3,1);
% ylim([-450,-50]);
% subplot(1,3,2);
% ylim([200,650]);
% subplot(1,3,3);
% ylim([-200,-165]);
% 
% subplot(1,3,1);
% ylim([400,2200]);
% subplot(1,3,2);
% ylim([2400,3100]);
% subplot(1,3,3);
% ylim([-400,0]);
% 
% subplot(1,3,1);
% ylim([-1200,-200]);
% subplot(1,3,2);
% ylim([650,1150]);
% subplot(1,3,3);
% ylim([-520,-400]);
addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/SSDGP/GPstuff-4.6'));
solar = 0;
concrete = 0;
mauna = 0;
pp = 1;

subset = 1;
m_values=[10,20,40,80,160,320];
numiter=1;
lb_table=zeros(numiter,length(m_values));
approx_ub_table=zeros(numiter,length(m_values));
ind =1;
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

if 1 == 0
num_workers=10;
num_iter = 10;
POOL=parpool('local',num_workers);
%m=80;
bic_table = zeros(num_iter,1);
gp_cell = cell(num_iter,1);
tic;
parfor i=1:num_iter
rng(i)
lik = lik_init(y);
gpcf_se1=se_init(x,y,1);
gpcf_se2=se_init(x,y,2);
gpcf_se3=se_init(x,y,3);
gpcf_se4=se_init(x,y,4);
gpcf=gpcf_prod('cf',{gpcf_se1,gpcf_se2,gpcf_se3,gpcf_se4});
%xu = datasample(x,m,1,'Replace',false);
[bic, gp] = gpfunction(x,y,gpcf,lik);
bic_table(i) = bic;
fprintf('iter %d : lb=%4.3f',i,bic);
gp_cell{i} = gp;
end
time = toc;
fprintf('time taken = %d',time);
save('/data/siris/not-backed-up/hkim/pp_ard.mat',bic_table,gp_cell,time);
delete(POOL);


for dim = 1:D
    fprintf('dim=%d \n',dim);
for m=m_values
    fprintf('m=%d \n',m);
for i=1:numiter
    %rng(i)
if solar
    %%% Initialise gp_var %%%
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_se2=se_init(x,y);
    gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
    if subset
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
end
if concrete
    %gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
    lik = lik_init(y);
    gpcf_lin4=lin_init(dim);
    fprintf('lin_sigma2 = %4.3f, signal_var = %4.3f \n',gpcf_lin4.coeffSigma2,lik.sigma2)
    gpcf_se1=se_init(x(:,1),y,1);
    gpcf_se2=se_init(x(:,2),y,2);
    gpcf_se4=se_init(x(:,4),y,4);
    gpcf_se7=se_init(x(:,7),y,7);
    gpcf_se8=se_init(x(:,8),y,8);
    gpcf1=gpcf_prod('cf',{gpcf_wn,gpcf_lin4});
    gpcf2=gpcf_prod('cf',{gpcf_se1,gpcf_se7});
    gpcf3=gpcf_prod('cf',{gpcf_se1,gpcf_se2,gpcf_se4});
    gpcf4=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se8});
    gpcf5=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se7,gpcf_se8,gpcf_lin4});
    if subset
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    % gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf_lin,'X_u', X_u);
end
if mauna
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    if subset
        X_u=datasample(x,m,1,'Replace',false); %random initialisation
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
end

% gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf',gpcf,'X_u', X_u);
% num_blocks=ceil(n/m);
% myind=cell(1,num_blocks);
% for block=1:num_blocks
%     myind{block} = (m*(block-1)+1):min(m*block,n);
% end
%gp_pic = gp_set('type', 'PIC', 'lik', lik, 'cf',gpcf,'X_u', X_u, 'tr_index',ind);
%gp_dtc = gp_set('type', 'DTC', 'lik', lik, 'cf',gpcf,'X_u', X_u);
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
%gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood');
%gp_pic = gp_set(gp_pic, 'infer_params', 'covariance+likelihood');
%gp_dtc = gp_set(gp_dtc, 'infer_params', 'covariance+likelihood');
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
warning('off','all');
gp_var_new=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
%w=gp_pak(gp_var);
%gp_fic=gp_unpak(gp_fic,w);
%gp_pic=gp_unpak(gp_pic,w);
%gp_dtc = gp_optim(gp_dtc,x,y,'opt',opt,'optimf',@fminscg);
% [energy,~]=gp_e([],gp_var_new,x,y);
% lb_table(i,ind) = -energy;
%gp_e([],gp_var,x,y)
%gp_e([],gp_fic,x,y)
%gp_e([],gp_pic,x,y)
%approx_ub_grad(w,gp_var,x,y)
%gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 

% if m==320
%     [gp_var,val]=approx_ub(gp_var,x,y,opt);
%     signal_var=gp_var.lik.sigma2;
%     K_mn=gp_cov(gp_var,X_u,x); K_mm=gp_trcov(gp_var,X_u);
%     L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
%     L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
%     A=L*L'+signal_var*eye(m);
%     L_naive=chol(A);
%     K_naive=L'*L;
%     n_cond=cond(K_naive + signal_var*eye(n));
%     fprintf('-val =%4.3f, sigma2=%4.8f, cond=%4.3f \n',...
%         -val, gp_var.lik.sigma2,n_cond);
%     approx_ub_table(i,ind)=-val;
% else
    %[gp_var_new,val_new]=approx_ub(gp_var_new,x,y,opt);
    %[gp_var,val]=approx_ub(gp_var,x,y,opt);
    %approx_ub_table(i,ind)=-min(val,val_new);
% end
%[gp_var_new,val]=minimax(gp_var,x,y,opt);
%w=gp_pak(gp_dtc);
%gp_pic=gp_unpak(gp_pic,w);
%gp_e([],gp_pic,x,y)
%fprintf('optim for worker %d done \n',i);
end
ind = ind + 1;
end
end

%ub = -temp;
%fprintf('lb = %4.3f, ub=%4.3f \n',lb,ub);

data = xlsread('/homes/hkim/Downloads/load.xlsx','zeros_copy');
new_data = data((data(:,2)~= 0),:);
times = new_data(:,1);
loads = new_data(:,2:end);

for i=1:20
    figure('units','normalized','outerposition',[0 0 1 1])
    plot(times,loads(:,i));
    str = ['Zone' i];
    title(str);
end
end

%load('/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/pp_skc_experiment_640m_1S.mat')
gp_var = kernel_top.gp_var;
lik = gp_var.lik;
gpcf = gp_var.cf{1};
gp=gp_set('lik',lik,'cf',gpcf);
p = length(gp_pak(gp)); % p is number of hyperparams
[n,~] = size(x); % n is n_data
[~, nll] = gp_e([],gp,x,y);
bic = -nll - p*log(n)/2

addpath(genpath(pwd));
range=[-2000,2500];
figure();
load('/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/pp_skc_experiment_160m_2S.mat')
subplot(1,3,1);
plot_skc(kernel_buffer_history,kernel_top,160,2);
xlim(range);
load('/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/pp_skc_experiment_320m_2S.mat')
subplot(1,3,2);
plot_skc(kernel_buffer_history,kernel_top,320,2)
xlim(range);
load('/data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/pp_skc_experiment_640m_1S.mat');
subplot(1,3,3);
plot_skc(kernel_buffer_history,kernel_top,640,2)
xlim(range);

%% import santa barbara hourly temperature data
fid=fopen('/homes/hkim/Downloads/temp_CA_Santa_Barbara_short.txt','r');
data = textscan(fid, '%d %s %s %d %d %f %f %f %f %f %f %f %*[^\n]','Delimiter',' ','MultipleDelimsAsOne',1);
fclose(fid);
idx = ((data{10} > -100) & (data{11} > -100) & (data{12} > -100));
temp_avg = data{10}(idx);
temp_max = data{11}(idx);
temp_min = data{12}(idx);
date_str = data{2}(idx);
time_str = data{3}(idx);

%% convert date and time to double %%
n = length(date_str);
time_converted = zeros(n,1);
for i = 1:n
    str = strcat(date_str{i},time_str{i});
    formatIn = 'yyyymmddHHMM';
    time_converted(i) = datenum(str,formatIn);
%    if mod(i,1000) == 0
%        fprintf('%d done', i);
%    end
end

%% import dover tidal data %%
temp_date_str={}; temp_time_str={}; temp_elevation=[];
for i=2012:2016 
    filename = strcat('/homes/hkim/Downloads/tidal/',int2str(i),'DOV.txt');
    fid = fopen(filename,'r');
    data = textscan(fid, '%s %s %s %f %*[^\n]','Delimiter',' ','MultipleDelimsAsOne',1,'HeaderLines',11);
    temp_date_str = [temp_date_str;data{2}];
    temp_time_str = [temp_time_str;data{3}];
    temp_elevation = [temp_elevation;data{4}];
end

% average over each hour
m = length(temp_elevation)/4;
date_str=cell(m,1);
time_str=cell(m,1);
elevation = zeros(m,1);
for i = 1:m;
    date_str{i} = temp_date_str{(i-1)*4+1};
    time_str{i} = temp_time_str{(i-1)*4+1};
    elevation(i) = mean(temp_elevation((i-1)*4+1:i*4));
end

idx = (elevation>-1);
date_str = date_str(idx);
time_str = time_str(idx);
elevation = elevation(idx);

%% convert date and time to double %%
n = length(date_str);
time_converted = zeros(n,1);
for i = 1:n
    str = strcat(date_str{i},time_str{i});
    formatIn = 'yyyy/mm/ddHH:MM:SS';
    time_converted(i) = datenum(str,formatIn);
%    if mod(i,1000) == 0
%        fprintf('%d done', i);
%    end
end

% get data between two big gaps
mydiff = diff(time_converted);
gap_idx = find((mydiff>10));
idx = (gap_idx(1)+2:gap_idx(2));
date_str = date_str(idx);
time_str = time_str(idx);
time_converted = time_converted(idx);
elevation = elevation(idx);

%% plot gefcom data %%
load gefcom.mat
[n,D] = size(times);
x = times; y=loads(:,1);
x_year = (x-min(x))/365 + 2004;
figure();
subplot(2,1,1);
plot(x_year,y);
xlabel('year')
ylabel('load(kW)')
xlim([2004,2008.5])
subplot(2,1,2);
x_day = x-min(x);
plot(x_day(1:7*24),y(1:7*24));
xlim([0,7])
ylim([0,5*(1e+4)])
xlabel('day in Jan 2004')
ylabel('load(kW)')


%% plot tidal data %%
load tidal.mat
n = length(time_converted);
x = time_converted; y = elevation;
x_year = (x-min(x))/365 + 2003;
% figure();
subplot(2,1,1);
plot(x_year,y);
xlabel('year')
ylabel('sea level elevation')
xlim([2003,2006.7])
ylim([0,8])
subplot(2,1,2);
x_day = x-min(x) + 1;
plot(x_day(1:56*24),y(1:56*24));
xlim([1,57])
ylim([0,8])
xlabel('days since Jan 2003')
ylabel('sea level elevation')

%%
function kernel_tree_plot(kernel_dict,m_values,y_lim)%,directory)
keys = kernel_dict.keys; 
for key_ind = 1:length(keys)
    key=keys{key_ind};
    kernel_cell = kernel_dict(key);
    lb = kernel_cell{1}; ub = kernel_cell{4}; ne = kernel_cell{5};
    figure();
    set(gca,'fontsize',18)
    hold on
    x_idx = 1:length(m_values);
    xlim([0.5,6.5])
    ylim(y_lim);
    plot(x_idx,ne*ones(size(m_values)),'LineWidth',2);
    midpt = 0.5*(ub + lb);
    errorbar(x_idx,midpt,lb,ub,'.','MarkerSize',0.1)
    plot(x_idx,ub,'LineWidth',3);
    plot(x_idx,lb,'LineWidth',1);
    set(gca,'XTick',[1 2 3 4 5 6]);
    set(gca,'XTickLabel',[10 20 40 80 160 320]);
    xlabel('m')
    %ylabel('negative energy')
    title(key)
    %legend('UB','fullGP','LB')
    hold off
    file_name=strcat(directory,key,'.fig');
    file_name = regexprep(file_name,'+','_plus_');
    file_name = regexprep(file_name,'*','_times_');
    saveas(fig,file_name)
end

end


