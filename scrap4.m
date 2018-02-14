%%% Script to extrapolate for tidal using kernel from skc

addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
load /data/anzu/not-backed-up/hkim/tidal_skc_experiment_640m_1S.mat

load tidal.mat
n = length(time_converted);
x = time_converted; y = elevation;
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
y = (y-y_mean)/y_std; %normalise y;

% find period to extrapolate out
x_diff = x(2)-x(1);
num_ext = 300;
x_missing = linspace(x(n),x(n)+x_diff*num_ext, num_ext)';

gp_var = kernel_top.gp_var;
[~,~,~,y_missing,y_missing_var] = gp_pred(gp_var,x,y,x_missing);
y_missing_std = sqrt(y_missing_var);

% plot missing values.
pre_ind = n-1000:n;
plot(x(pre_ind),y(pre_ind),'b')
hold on
fill([x_missing;flipud(x_missing)],[y_missing-y_missing_std;flipud(y_missing+y_missing_std)],[1.,.9,.9],'linestyle','none');
plot(x_missing,y_missing,'--r')
hold off