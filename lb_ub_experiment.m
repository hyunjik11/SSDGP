addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/SSDGP/GPstuff-4.6'));

num_workers=10;
%POOL=parpool('local',num_workers);

warning('off','all');
solar = 1;
concrete = 0;
mauna = 0;

num_iter=10;
m_values=[10,20,40,80,160,320];

%%%% Initialise data %%%%
if solar
    load solar.mat
    x=X; Y=y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
    %%% Initialise gp_var %%%
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_se2=se_init(x,y);
    gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
end
if concrete
    load concrete.mat
    x=X; Y=y;
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std;
    gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
    lik = lik_init(y);
    gpcf_lin4=lin_init(4);
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
    gpcf = gpcf_sum('cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5});
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
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    gpcf = gpcf_sum('cf',{gpcf1,gpcf2,gpcf3}); 
end
%%%%%%%%%%%%%%%%%%%

nm=length(m_values);
lb_table = zeros(num_iter,nm); % stores lb for all iter
ub_table = zeros(num_iter,nm); % stores ub for all iter
gp_table = zeros(num_iter,1); % stores bic for all iter
gp_var_cell = cell(num_iter,nm); % stores lb gp_var for all iter
gp_ub_cell = cell(num_iter,nm); % stores ub gp_var for all iter
gp_cell = cell(num_iter,1); % stores gp for all iter
idx_cell = cell(num_iter,nm); % stores indices for ind pts for all iter
idx_u = 1:n; %idx_u used to store indices of subset for best LB for previous m
[gpcf_best,lik_best] = reinitialise_kernel(gpcf,x,y); %temporary initialisation

for i = 1:nm
    m = m_values(i);
    parfor iter = 1:num_iter % change to parfor for parallel
        rng(iter);
        warning('off','all');
        
        %%% optim for gp
        if i==1
            [gpcf_gp,lik_gp] = reinitialise_kernel(gpcf,x,y);
            [gp_table(iter), gp_cell{iter}] = gpfunction(x,y,gpcf_gp,lik_gp);
        end
        %%% optim for lb
        if i==1 || iter <= 0.8*num_iter % use rand init of hyp for all iter of first m & 4/5 of iters for other m's
            [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false);
            [gpcf_lb, lik_lb] = reinitialise_kernel(gpcf,x,y);
            [lb_table(iter,i),gp_var] = lbfunction(x,y,xu,gpcf_lb,lik_lb);
            gp_var_cell{iter,i} = gp_var;
            [ub_table(iter,i),gp_ub_cell{iter,i}] = ub(gp_var,x,y);
        else % for 1/5 of iter, keep optimal ind pts from previous m, and also keep the hyp
            weights = 1e-10*ones(1,n); %weights for sampling
            weights(idx_u)=1; %make sure samples idx_u are included
            [xu,idx_cell{iter,i}] = datasample(x,m,1,'Replace',false,'Weights',weights);
            [lb_table(iter,i),gp_var] = lbfunction(x,y,xu,gpcf_best,lik_best);
            gp_var_cell{iter,i} = gp_var;
            [ub_table(iter,i),gp_ub_cell{iter,i}] = ub(gp_var,x,y);
        end
    end
    [~,ind] = max(lb_table(:,i));
    idx_u = idx_cell{ind,i}; %indices of subset for best LB
    
    %%% get hyp from best LB
    gp_var_best = gp_var_cell{ind,i};
    gpcf_best = gp_var_best.cf{1};
    lik_best = gp_var_best.lik;
    
    fprintf('m=%d done \n',m);
end

lower = max(lb_table);
upper = max(ub_table);
bic = max(gp_table);

figure();
hold on
plot(1:nm,upper,'r');
plot(1:nm,bic*ones(size(m_values)),'g');
plot(1:nm,lower,'b');
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('UB','GP','LB')
ylabel('BIC')
xlabel('m')
hold off
%saveas(gcf,'plots/approx_ub_concrete.fig')








