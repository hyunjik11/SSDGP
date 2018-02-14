% figure();
% hold on
% m_values=[1,2,3,4,5,6];
% %nll = -2029.864; % mauna full GP
% %nll = 83.864; % solar full GP
% %nll = 248.82; % concrete full GP
% %ne = -89.3388; % solar full GP
% %ne = 2002.6; % mauna full GP
% ne = -294.1799; % concrete full GP
% plot(m_values,ne*ones(size(m_values)),'b');
% scatter(reshape(repmat(m_values,10,1),60,1),reshape(lb_table,60,1),'gx');
% plot(m_values,max(lb_table),'g');
% scatter(reshape(repmat(m_values,10,1),60,1),reshape(approx_ub_table,60,1),'rx');
% plot(m_values,max(approx_ub_table),'r');
% %plot(m_values,max(ub_fic_table));
% %plot(m_values,max(ub_pic_table));
% set(gca,'XTick',[1 2 3 4 5 6]);
% set(gca,'XTickLabel',[10 20 40 80 160 320]);
% legend('fullGP','VAR LB','highest LB','UB', 'highest UB')
% ylabel('negative energy')
% xlabel('m')
% hold off
% saveas(gcf,'plots/approx_ub_concrete.fig')
% 
% temp = kernel_dict('SE');
% gpcf=temp{2}{6};
% lik=temp{3}{6};
% gp=gp_set('lik',lik,'cf',gpcf);
% energy = gp_e([],gp,x,y);

% num_iter=20;
% %POOL=parpool('local',num_iter);
% m_values=[10,20,40,80,160,320];
% ne_table=zeros(num_iter,length(m_values));
% gp_var_cell=cell(num_iter,length(m_values));
% for m_iter = 1:length(m_values);
%     m=m_values(m_iter);
%     parfor i=1:num_iter
%         warning('off','all');
%         gpcf=gpcf_prod('cf',{per_init(x,y),lin_init()});
%         lik = lik_init(y);
%         xu = datasample(x,m,1,'Replace',false);
%         gp_var = gp_set('type','VAR','cf',gpcf,'lik',lik,'X_u',xu);
%         opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
%         gp_var=gp_optim(gp_var,x,y,'opt',opt);
%         energy=gp_e([],gp_var,x,y);
%         ne_table(i,m_iter)=-energy;
%         gp_var_cell{i,m_iter}= gp_var;
%     end
%     fprintf('m=%d done \n',m);
% end
% 
% hold on
% xlim([0.5,6.5])
% ylim([-1000,1800]);
% plot(1:6,max(ne_table));
% scatter(reshape(repmat(1:6,num_iter,1),6*num_iter,1),reshape(ne_table,6*num_iter,1),'gx');
% hold off
% %delete(POOL)

%%% Script to interpolate the missing weeks for gefcom data using kernel
%%% from skc
addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
% load /data/greyheron/not-backed-up/oxwasp/oxwaspor/hkim/gefcom_skc_experiment_160m_2S.mat
load /data/greywagtail/not-backed-up/oxwasp/oxwaspor/hkim/gefcom_skc_experiment_320m_1S.mat
load gefcom.mat
[n,D] = size(times);
x = times; y=loads(:,1);
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
y = (y-y_mean)/y_std; %normalise y;

% find missing periods
x_diff = x(2)-x(1);
ind = 1;
gap_begin = zeros(8,1);
gap_begin_ind = zeros(8,1);
gap_end = zeros(8,1);
for i = 1:n-1
    if x(i+1)-x(i) > x_diff + 1e-5
        gap_begin(ind) = x(i);
        gap_end(ind) = x(i+1);
        gap_begin_ind(ind) = i;
        ind = ind + 1;
    end
end

num_pts = 167+2;
x_missing = zeros(8*num_pts,1);
for i = 1:8
    beg = gap_begin(i);
    fin = gap_end(i);
%     num_pts = round((fin-beg)/x_diff);
    x_missing((i-1)*num_pts+1:i*num_pts) = linspace(beg,fin, num_pts);
end

gp_var = kernel_top.gp_var;
[~,~,~,y_missing,y_missing_var] = gp_pred(gp_var,x,y,x_missing);
y_missing_std = sqrt(y_missing_var);

% plot missing values.
i=5;
pre_ind = gap_begin_ind(i)-num_pts:gap_begin_ind(i);
post_ind = gap_begin_ind(i)+1:gap_begin_ind(i)+num_pts;
missing_ind = (i-1)*num_pts+1:i*num_pts;
plot(x(pre_ind),y(pre_ind),'b')
hold on
plot(x(post_ind),y(post_ind),'b')
fill([x_missing(missing_ind);flipud(x_missing(missing_ind))],...
    [y_missing(missing_ind)-y_missing_std(missing_ind);...
    flipud(y_missing(missing_ind)+y_missing_std(missing_ind))],[1. .9 .9],'linestyle','none');
plot(x_missing(missing_ind),y_missing(missing_ind),'--r')
hold off












