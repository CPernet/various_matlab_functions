% variable impulse vs variable epoch model
% this simulation is designed to explore the effect
% of variable epoch lenghth not only on a voxel where
% this effect exist but also on a voxel where it doesn't
% and we look if there is a better trade off in terms in
% modelling so that we can detect well both types and still
% make inferences about which one has variable epoch using
% derivative information
% ------------------------
% Cyril Pernet 23-07-2014

% RT distribution -- we take here RT for decisions which are 'easy'
% ie relatively fast, in the order of 1 to 2 sec
N = 500;
RT_condition1 = NaN(1,N);
RT_condition2 = NaN(1,N);
 for event = 1:N % draw 100 random data from a gamma distribution
    if mod(event,2) == 0
    RT_condition1(event) = gamrnd(5,10);
    else
    RT_condition2(event) = gamrnd(5,10) +20; % cond1/cond2 differ 
    end
end
RT_condition1(isnan(RT_condition1))=[];
RT_condition2(isnan(RT_condition2))=[];
figure; 
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized', ...
    'outerposition',[0 0 1 1])
subplot(1,2,1);
[h1,x1]=hist(RT_condition1,20);
[h2,x2]=hist(RT_condition2,20);
bar(x1, h1, 'r'); hold on; bar(x2, h2);
h=findobj(gca, 'Type', 'patch');
set (h, 'FaceAlpha', 0.7); grid on
title('populations to draw RT from','FontSize',14);
% rescale to maximum 6 
Scale = max([RT_condition1 RT_condition2])/6; 
RT_condition1 = RT_condition1./Scale;
RT_condition2 = RT_condition2./Scale;
% cut data from 3 and keep only 10 data points
RT_condition1(RT_condition1<3) = [];
RT_condition1 = sort(RT_condition1);
RT_condition1 = RT_condition1(1:10);
RT_condition2(RT_condition2<1) = [];
RT_condition2 = sort(RT_condition2,'descend');
RT_condition2 = RT_condition2(1:10);
% check we have nice differences between conditions
subplot(1,2,2);boxplot([RT_condition1' RT_condition2'])
[H,P,CI,STATS] = ttest2(RT_condition1,RT_condition2);
D = mean(RT_condition1)-mean(RT_condition2);
title(sprintf('difference=%g CI [%g %g] \n t(%g)=%g p=%g' ...
    ,D,CI(1),CI(2),STATS.df,STATS.tstat,P),'FontSize',14);
grid on; 

%% BOLD simulation
% now make 2 voxels with 2 conditions
% voxel 1 duration = 0 
% voxel 2 duration = RT 
% note that for ease of interpretation 
% I work using the super-sampled design 

% hemodynamic response function
% hrf model using SPM function 
% --------------------------- 
xBF.dt = 0.5;
xBF.name = 'hrf (with time and dispersion derivative)'; 
xBF.length = 32; % over a 20 sec window 
xBF.order = 1; 
xBF.T = 30;
xBF = spm_get_bf(xBF);
xBF.UNITS = 'secs';

% make different BOLD data
scale1 = detrend(RT_condition1,'constant').*100;
scale1 = abs(scale1 ./ min(scale1)) ; % stimulus intensity 
scale2 = detrend(RT_condition2,'constant').*100;
scale2 = abs(scale2 ./ min(scale2));
onsets = [1 31 81 93 161 201 218 291 321 361];
duration = round(mean([RT_condition1 RT_condition2])*2); 
Y1 = zeros(500,1); % condition 1, with 0ms duration
Y2 = zeros(500,1); % condition 2, with 0ms duration 
                   % but we have intensity differences proportional to RT
Y3 = zeros(500,1); % condition 2, with 0ms duration
Y4 = zeros(500,1); % condition 2, with RT duration
X1 = zeros(500,1); % model of event 
X2 = zeros(500,1); % model of blocks of mean RT
X3 = zeros(500,1); % model of blocks of RT condition 1
X4 = zeros(500,1); % model of blocks of RT condition 2
for i=1:10
    Y1(onsets(i)) = scale1(i); % finite impulse of different intensities
    Y2(onsets(i)) = scale2(i) ; 
    X1(onsets(i)) = 1; % design matrix with stim onset
    X2(onsets(i):(onsets(i)+duration-1)) = 1; % design matrix with stim onset
    stop = onsets(i) + round(RT_condition1(i)*2) -1;
    Y3(onsets(i):stop) = 1; % epoch = RT
    X3(onsets(i):stop) = 1;
    stop = onsets(i) + round(RT_condition2(i)*2) -1;
    Y4(onsets(i):stop) = 1; % epoch = RT
    X4(onsets(i):stop) = 1;
end 
Y1 = conv(Y1+randn(500,1)./10,xBF.bf(:,1)); Y1 = Y1(1:400)+100;
Y2 = conv(Y2+randn(500,1)./10,xBF.bf(:,1)); Y2 = Y2(1:400)+100;
Y3 = conv(Y3,xBF.bf(:,1))+conv(rand(500,1),xBF.bf(:,1)); Y3 = Y3(1:400)+100;
Y4 = conv(Y4,xBF.bf(:,1))+conv(rand(500,1),xBF.bf(:,1)); Y4 = Y4(1:400)+100;
figure; subplot(2,1,1);
plot(Y1,'LineWidt',2); hold on; plot(Y2,'r','LineWidt',2);
grid on; title('Neural Intensity difference model'); subplot(2,1,2);
plot(Y3,'LineWidt',2); hold on; plot(Y4,'k','LineWidt',2); 
                                 % what is evident from this is that the 
                                 % variable epoch not only give different 
                                 % intensities but also dynamic
grid on; title('Neural duration difference model');

%% data modelling
% data modelling
% for simplicity - assume Y1/Y2 are the same voxel and we fit X to it
% same for Y3/Y4 ; it just easier to see data fit when done individually
% the mean RT blocks + paramtetric model comes from Munford & Poldrack 2014
% Adjusting mean activation for reaction time effects in BOLD fMRI
% which was presented ath the OHBM conference

%% model data for a voxel with intensity change
Y = [Y1 ; Y2];
SStotal = norm(Y-mean(Y)).^2;
[~,~,~,stat]=ttest2(Y2,Y1);

% -> clearly for a voxel where only intensity changes, model 2 doesn't work
% -> model 3 (hrf + parametric) is a priori the best
% -> using mean RT will not work (although here is is mean to roughly
% follow by using intensities that are correlated with duration)
% -> the question is how mean RT + parametric using BF to accomodate the
% mismatch is going to behave

% model 1: GLM with FIR convolved by the hfr only
x1 = conv(X1,xBF.bf(:,1)); x1 = x1(1:400)+100;
X = [[x1;zeros(400,1)] [zeros(400,1);x1] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 1 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con = 2*(1-spm_Tcdf(T_con, (size(Y,1)-rank(X)))); 
figure ; set(gcf,'Color','w','InvertHardCopy','off', ...
    'units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y1,'LineWidth',2); hold on
plot(Yhat(1:400),'--','LineWidth',2); 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('fixed event model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))

% model 2: GLM with RT convolved by the hfr only (= variable epoch model)
x1 = conv(X3,xBF.bf(:,1)); x1 = x1(1:400)+100;
x2 = conv(X4,xBF.bf(:,1)); x2 = x2(1:400)+100;
X = [[x1;zeros(400,1)] [zeros(400,1);x2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 1 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con = 2*(1-spm_Tcdf(T_con, (size(Y,1)-rank(X)))); 
subplot(3,1,2); 
plot(Y1,'LineWidth',2); hold on
plot(Yhat(1:400),'--','LineWidth',2); 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('variable epoch model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))

% model 3: GLM with FIR + PM regressors convolved by the hfr only 
% (= variable impulse model)
x1 = conv(X1,xBF.bf(:,1)); x1 = x1(1:400)+100;
x2 = X1; x2(X1==1) = detrend(RT_condition1,'constant'); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM1 = spm_orth([x1 x2]); % x with parameteric modulation for Y1
x2 = X1; x2(X1==1) = detrend(RT_condition2,'constant'); 
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM2 = spm_orth([x1 x2]); % x with parameteric modulation for Y2
X = [[XPM1;zeros(400,2)] [zeros(400,2);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 1 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
subplot(3,1,3); 
plot(Y1,'LineWidth',2); hold on; 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('event + parametric model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g ',R2,Difference_real_model,stat.tstat,T_con))

% let's gain a little insight of how the parametric model works
figure ; 
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(2,1,1);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 5])*b([1 5]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y1','hrf regressor','hrf+modulation')
grid on; subplot(2,1,2);
plot(Y2,'LineWidth',2); hold on
yhat = X(:,[3 5])*b([3 5]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y2','hrf regressor','hrf+modulation')
grid on; 

% model 4: GLM with FIR + PM regressors convolved by basis functions
x11 = conv(X1,xBF.bf(:,1)); x11 = x11(1:400)+100;
x12 = conv(X1,xBF.bf(:,2)); x12 = x12(1:400)+100;
x13 = conv(X1,xBF.bf(:,3)); x13 = x13(1:400)+100;
x2 = X1; x2(X1==1) = detrend(RT_condition1,'constant'); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM1 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y1
x2 = X1; x2(X1==1) = detrend(RT_condition2,'constant'); 
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM2 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y2
X = [[XPM1;zeros(400,6)] [zeros(400,6);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 0 0 0 0 1 0 0 0 0 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
% note that here really we need a F contrast spanning the BF
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y1,'LineWidth',2); hold on; 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('event*BF with parametric*BF model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 2 3 13])*b([1 2 3 13]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y1','BF regressor','BF+modulation')
grid on; subplot(3,1,3);
plot(Y2,'LineWidth',2); hold on
yhat = X(:,[7 8 9 13])*b([7 8 9 13]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y2','BF regressor','BF+modulation')
grid on;

% model 5: GLM with mean RT + PM regressors convolved by the hfr only
x1 = conv(X2,xBF.bf(:,1)); x1 = x1(1:400)+100;
PM = repmat(detrend(RT_condition1,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM1 = spm_orth([x1 x2]); % x with parameteric modulation for Y1
PM = repmat(detrend(RT_condition2,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM2 = spm_orth([x1 x2]); % x with parameteric modulation for Y2
X = [[XPM1;zeros(400,2)] [zeros(400,2);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 1 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y1,'LineWidth',2); hold on; 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('mean RT + parametric model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g ',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 5])*b([1 5]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y1','hrf regressor','hrf+modulation')
grid on; subplot(3,1,3);
plot(Y2,'LineWidth',2); hold on
yhat = X(:,[3 5])*b([3 5]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y2','hrf regressor','hrf+modulation')
grid on; 

% model 6: GLM with mean RT + PM regressors convolved by basis functions 
% adding BF will accomodate well the mismatched durations -- note there is 
% no point to comparing to a fixed event model with BF since there is no
% time shift in those 2 time series
x11 = conv(X2,xBF.bf(:,1)); x11 = x11(1:400)+100;
x12 = conv(X2,xBF.bf(:,2)); x12 = x12(1:400)+100;
x13 = conv(X2,xBF.bf(:,3)); x13 = x13(1:400)+100;
PM = repmat(detrend(RT_condition1,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM1 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y1
PM = repmat(detrend(RT_condition2,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM2 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y2
X = [[XPM1;zeros(400,6)] [zeros(400,6);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 0 0 0 0 1 0 0 0 0 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
% note that here really we need a F contrast spanning the BF
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y1,'LineWidth',2); hold on; 
plot(Y2,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y1-Y2);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('mean RT*BF + parametric*BF model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 2 3 13])*b([1 2 3 13]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y1','BF regressor','BF+modulation')
grid on; subplot(3,1,3);
plot(Y2,'LineWidth',2); hold on
yhat = X(:,[7 8 9 13])*b([7 8 9 13]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y2','BF','BF+modulation')
grid on;

% assuming we model data using this model 6 -- how can we know the effect
% is due to differences in neural intensity
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(2,1,1);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 13])*b([1 13]);
plot(yhat(1:400),'g','LineWidth',2);
yhat = X(:,[1 2 13])*b([1 2 13]);
plot(yhat(1:400),'c','LineWidth',2);
yhat = X(:,[1 3 13])*b([1 3 13]);
plot(yhat(1:400),'r','LineWidth',2);
legend('Y1','mean RT regressor','mean RT with time deriv',...
    'mean RT with disp deriv')
grid on; 

% -> the mean RT creates time delays well compensated by the 1st derivative
% in model 4, the time deriv is significant too - is that a problem? 
C = [0 1 0 0 0 0 0 1 0 0 0 0 0]; % contrast for time derivative
T_con1 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con1 = 2*(1-spm_Tcdf(T_con1, (size(Y,1)-rank(X))));
C = [0 0 1 0 0 0 0 0 1 0 0 0 0]; % contrast for disp derivative
T_con2 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con2 = 2*(1-spm_Tcdf(T_con2, (size(Y,1)-rank(X))));
title(sprintf('model for the mean RT regressor \n time deriv t=%g p-%g disper deriv t=%g p=%g',T_con1,p_con1,T_con2,p_con2))

subplot(2,1,2);
plot(Y1,'LineWidth',2); hold on
yhat = X(:,[1 4 13])*b([1 4 13]);
plot(yhat(1:400),'g','LineWidth',2);
yhat = X(:,[1 2 4 5 13])*b([1 2 4 5 13]);
plot(yhat(1:400),'c','LineWidth',2);
yhat = X(:,[1 3 4 6 13])*b([1 3 4 6 13]);
plot(yhat(1:400),'r','LineWidth',2);
legend('Y1','mean RT+PM regressor','mean RT+PM with time deriv','mean RT+PM with disp deriv')
grid on; 

% --> PM derivatives don't add much - they are significant but we can't
% interprete much from this
C = [0 0 0 0 1 0 0 0 0 0 1 0 0]; % contrast for PM time derivative
T_con1 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con1 = 2*(1-spm_Tcdf(T_con1, (size(Y,1)-rank(X))));
C = [0 0 0 0 0 1 0 0 0 0 0 1 0]; % contrast for PM disp derivative
T_con2 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con2 = 2*(1-spm_Tcdf(T_con2, (size(Y,1)-rank(X))));
title(sprintf('model for the mean RT+PM regressor \n PM time deriv t=%g p-%g PM disper deriv t=%g p=%g',T_con1,p_con1,T_con2,p_con2))


%% model data for a voxel with duration change
Y = [Y3 ; Y4];
SStotal = norm(Y-mean(Y)).^2;
[h,p,ci,stat]=ttest2(Y4,Y3);

% model 1: GLM with FIR convolved by the hfr only
x1 = conv(X1,xBF.bf(:,1)); x1 = x1(1:400)+100;
X = [[x1;zeros(400,1)] [zeros(400,1);x1] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 1 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con = 2*(1-spm_Tcdf(T_con, (size(Y,1)-rank(X)))); 
figure ; set(gcf,'Color','w','InvertHardCopy','off', 'units',...
    'normalized','outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y3,'LineWidth',2); hold on
plot(Yhat(1:400),'--','LineWidth',2); 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('fixed event model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))

% model 2: GLM with RT convolved by the hfr only (= variable epoch model)
x1 = conv(X3,xBF.bf(:,1)); x1 = x1(1:400)+100;
x2 = conv(X4,xBF.bf(:,1)); x2 = x2(1:400)+100;
X = [[x1;zeros(400,1)] [zeros(400,1);x2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 1 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con = 2*(1-spm_Tcdf(T_con, (size(Y,1)-rank(X)))); 
subplot(3,1,2); 
plot(Y3,'LineWidth',2); hold on
plot(Yhat(1:400),'--','LineWidth',2); 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('variable epoch model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))

% model 3: GLM with FIR + PM regressors convolved by the hfr only 
% (= variable impulse model)
x1 = conv(X1,xBF.bf(:,1)); x1 = x1(1:400)+100;
x2 = X1; x2(X1==1) = detrend(RT_condition1,'constant'); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM1 = spm_orth([x1 x2]); % x with parameteric modulation for Y1
x2 = X1; x2(X1==1) = detrend(RT_condition2,'constant'); 
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM2 = spm_orth([x1 x2]); % x with parameteric modulation for Y2
X = [[XPM1;zeros(400,2)] [zeros(400,2);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 1 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
subplot(3,1,3); 
plot(Y3,'LineWidth',2); hold on; 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'g','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('event + parametric model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g ',R2,Difference_real_model,stat.tstat,T_con))

% let's gain a little insight of how the parametric model works
figure ; 
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(2,1,1);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 5])*b([1 5]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y3','hrf regressor','hrf+modulation')
grid on; subplot(2,1,2);
plot(Y4,'LineWidth',2); hold on
yhat = X(:,[3 5])*b([3 5]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y4','hrf regressor','hrf+modulation')
grid on; 

% model 4: GLM with FIR + PM regressors convolved by basis functions
x11 = conv(X1,xBF.bf(:,1)); x11 = x11(1:400)+100;
x12 = conv(X1,xBF.bf(:,2)); x12 = x12(1:400)+100;
x13 = conv(X1,xBF.bf(:,3)); x13 = x13(1:400)+100;
x2 = X1; x2(X1==1) = detrend(RT_condition1,'constant'); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM1 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y1
x2 = X1; x2(X1==1) = detrend(RT_condition2,'constant'); 
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM2 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y2
X = [[XPM1;zeros(400,6)] [zeros(400,6);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 0 0 0 0 1 0 0 0 0 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
% note that here really we need a F contrast spanning the BF
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y3,'LineWidth',2); hold on; 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('event*BF with parametric*BF model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 2 3 13])*b([1 2 3 13]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y3','BF regressor','BF+modulation')
grid on; subplot(3,1,3);
plot(Y4,'LineWidth',2); hold on
yhat = X(:,[7 8 9 13])*b([7 8 9 13]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y4','BF regressor','BF+modulation')
grid on;

% model 5: GLM with mean RT + PM regressors convolved by the hfr only
x1 = conv(X2,xBF.bf(:,1)); x1 = x1(1:400)+100;
PM = repmat(detrend(RT_condition1,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM1 = spm_orth([x1 x2]); % x with parameteric modulation for Y1
PM = repmat(detrend(RT_condition2,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x2 = conv(x2,xBF.bf(:,1)); x2 = x2(1:400)+100;
XPM2 = spm_orth([x1 x2]); % x with parameteric modulation for Y2
X = [[XPM1;zeros(400,2)] [zeros(400,2);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 1 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y3,'LineWidth',2); hold on; 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('mean RT + parametric model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g ',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 5])*b([1 5]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y3','hrf regressor','hrf+modulation')
grid on; subplot(3,1,3);
plot(Y4,'LineWidth',2); hold on
yhat = X(:,[3 5])*b([3 5]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y4','hrf regressor','hrf+modulation')
grid on; 

% model 6: GLM with mean RT + PM regressors convolved by basis functions 
% adding BF will accomodate well the mismatched durations -- note there is 
% no point to comparing to a fixed event model with BF since there is no
% time shift in those 2 time series
x11 = conv(X2,xBF.bf(:,1)); x11 = x11(1:400)+100;
x12 = conv(X2,xBF.bf(:,2)); x12 = x12(1:400)+100;
x13 = conv(X2,xBF.bf(:,3)); x13 = x13(1:400)+100;
PM = repmat(detrend(RT_condition1,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM1 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y1
PM = repmat(detrend(RT_condition2,'constant')',[1,duration])';
x2 = X2; x2(X2==1) = PM(:); % mean center RT
x21 = conv(x2,xBF.bf(:,1)); x21 = x21(1:400)+100;
x22 = conv(x2,xBF.bf(:,2)); x22 = x22(1:400)+100;
x23 = conv(x2,xBF.bf(:,3)); x23 = x23(1:400)+100;
XPM2 = spm_orth([x11 x12 x13 x21 x22 x23]); % x with PM for Y2
X = [[XPM1;zeros(400,6)] [zeros(400,6);XPM2] ones(800,1)];
b = pinv(X)*Y; 
Yhat = X*b;
SSeffect = norm(Yhat-mean(Yhat)).^2;
R2 = SSeffect / SStotal;
P = X*pinv(X); % H matrix 
R = eye(size(Y,1)) - P; 
variance = ((R*Y)'*(R*Y)) / (size(Y,1)-rank(X));
C = [-1 0 0 0 0 0 1 0 0 0 0 0 0]; % contrast for condition 2 > condition 1
T_con = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
% note that here really we need a F contrast spanning the BF
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(3,1,1); 
plot(Y3,'LineWidth',2); hold on; 
plot(Y4,'r','LineWidth',2); 
plot(Yhat(1:400),'--','LineWidth',2);
plot(Yhat(401:end),'--r','LineWidth',2); grid on; axis tight
D = mean(Y3-Y4);
Dhat = mean(Yhat(1:400)-Yhat(401:end));
Difference_real_model = D-Dhat;
title(sprintf('mean RT*BF + parametric*BF model: R2=%g diff data model =%g C1 vs C2 t real=%g t model=%g',R2,Difference_real_model,stat.tstat,T_con))
subplot(3,1,2);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 2 3 13])*b([1 2 3 13]);
plot(yhat(1:400),'k','LineWidth',2);
plot(Yhat(1:400),'r','LineWidth',2);
legend('Y3','BF regressor','BF+modulation')
grid on; subplot(3,1,3);
plot(Y4,'LineWidth',2); hold on
yhat = X(:,[7 8 9 13])*b([7 8 9 13]);
plot(yhat(401:end),'k','LineWidth',2);
plot(Yhat(401:end),'r','LineWidth',2);
legend('Y4','BF','BF+modulation')
grid on;

% assuming we model data using this model 6 -- how can we know the effect
% is due to differences in neural intensity
figure
set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized',...
    'outerposition',[0 0 1 1])
subplot(2,1,1);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 13])*b([1 13]);
plot(yhat(1:400),'g','LineWidth',2);
yhat = X(:,[1 2 13])*b([1 2 13]);
plot(yhat(1:400),'c','LineWidth',2);
yhat = X(:,[1 3 13])*b([1 3 13]);
plot(yhat(1:400),'r','LineWidth',2);
legend('Y3','mean RT regressor','mean RT with time deriv',...
    'mean RT with disp deriv')
grid on; 

% -> the mean RT creates time delays well compensated by the 1st derivative
% in model 4, the time deriv is significant too - is that a problem? 
C = [0 1 0 0 0 0 0 1 0 0 0 0 0]; % contrast for time derivative
T_con1 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con1 = 2*(1-spm_Tcdf(T_con1, (size(Y,1)-rank(X))));
C = [0 0 1 0 0 0 0 0 1 0 0 0 0]; % contrast for disp derivative
T_con2 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con2 = 2*(1-spm_Tcdf(T_con2, (size(Y,1)-rank(X))));
title(sprintf('model for the mean RT regressor \n time deriv t=%g p-%g disper deriv t=%g p=%g',T_con1,p_con1,T_con2,p_con2))

subplot(2,1,2);
plot(Y3,'LineWidth',2); hold on
yhat = X(:,[1 4 13])*b([1 4 13]);
plot(yhat(1:400),'g','LineWidth',2);
yhat = X(:,[1 2 4 5 13])*b([1 2 4 5 13]);
plot(yhat(1:400),'c','LineWidth',2);
yhat = X(:,[1 3 4 6 13])*b([1 3 4 6 13]);
plot(yhat(1:400),'r','LineWidth',2);
legend('Y3','mean RT+PM regressor','mean RT+PM with time deriv',...
    'mean RT+PM with disp deriv')
grid on; 

% --> PM derivatives don't add much - they are significant but we can't
% interprete much from this
C = [0 0 0 0 1 0 0 0 0 0 1 0 0]; % contrast for PM time derivative
T_con1 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con1 = 2*(1-spm_Tcdf(T_con1, (size(Y,1)-rank(X))));
C = [0 0 0 0 0 1 0 0 0 0 0 1 0]; % contrast for PM disp derivative
T_con2 = (C*b) ./ sqrt(variance.*(C*pinv(X'*X)*C')); % T value
p_con2 = 2*(1-spm_Tcdf(T_con2, (size(Y,1)-rank(X))));
title(sprintf('model for the mean RT+PM regressor \n PM time deriv t=%g p-%g PM disper deriv t=%g p=%g',T_con1,p_con1,T_con2,p_con2))
