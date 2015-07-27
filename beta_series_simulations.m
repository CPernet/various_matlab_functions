%% How to model data for beta series correlations
% Typically, one makes a design matrix with n trials to get n+1 beta maps
% (the n trials plus the model adjusted mean). Next, from those n maps, one
% reads the data in a ROI to create a vector a beta values. Finally, that 
% vector is used as input in a new deisgn matrix and ones regress on the n
% beta maps, ie that tells us from the ROI, which other voxels show that 
% same trial to trial variance - which is an indication of functional 
% connectivity (ie when this voxel changes the other one changes too).

%%% *What is this illustration about*
% the goal is to show that, if differences exist between conditions
% then it must be accounted for in the analysis ; this is what is done
% in PPI analyses and this must be computed as well for beta series

%% _let's simulate dome data_
%%% generate 50 beta for 3 conditions and 3 ROI
A = randn(150,3);

% for the sake of the simulation make sure the ROI correlations are close
% to zero by applying a Gram-Smidt orthogonalization 
% <http://web.mit.edu/18.06/www/Essays/gramschmidtmat.pdf>

[m,n]=size(A);
Q=zeros(m,n);
R=zeros(n,n);
for j=1:n
    v=A(:,j);
    for i=1:j-1
        R(i,j)=Q(:,i)'*A(:,j);
        v=v-R(i,j)*Q(:,i);
    end
R(j,j)=norm(v);
Q(:,j)=v/R(j,j);
end

%% conduct a beta series correlations as a single condition
% if no correlation, p non significant

Q = (Q.*10) + randn(150,3)./10; % rescale and add noise
Y = Q(:,[2 3]); % the data to regress on (would be other voxels of the brain)
X = [Q(:,1) ones(150,1)]; % our ROI (ie we extract betas per trial in that ROI);
figure; tmp = zscore(X(:,1)); imagesc([tmp/max(tmp) X(:,2)]); 
colormap('gray'); title('Design with beta only','FontSize',14);
Beta = pinv(X)*Y; % now solve the regression
Yhat = X*Beta; Res = Y-Yhat;
var = diag(Res'*Res) / (length(Y)-rank(X));

figure; subplot(1,2,1); scatter(Q(:,1),Y(:,1),50,'k'); hold on
plot(Q(:,1),Yhat(:,1),'r','LineWidth',3); grid on
t = Beta(1,1) / sqrt(var(1)*([1 0]*inv(X'*X)*[1 0]'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(1,1),t,p);
title(mytitle,'FontSize',14);

subplot(1,2,2); scatter(Q(:,1),Y(:,2),50,'k'); hold on
plot(Q(:,1),Yhat(:,2),'r','LineWidth',3); grid on
t = Beta(1,2) / sqrt(var(2)*([1 0]*inv(X'*X)*[1 0]'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(1,1),t,p);
title(mytitle,'FontSize',14);

%% conduct a beta series correlations with 3 conditions
% imagine that, on average, cond1 > cond2 > cond3 in ROI 1 and ROI 2 
% but all equal in ROI 3

Q(51:100,[1 2]) = Q(51:100,[1 2]) + 2;
Q(101:150,[1 2]) = Q(101:150,[1 2]) + 4;
 
%%% _let's now recompute without the conditions_
Y = Q(:,[2 3]); 
X = [Q(:,1) ones(150,1)]; 
figure; tmp = zscore(X(:,1)); imagesc([tmp/max(tmp) X(:,2)]); 
colormap('gray'); title('Design with beta only','FontSize',14);
Beta = pinv(X)*Y;
Yhat = X*Beta; Res = Y-Yhat;
var = diag(Res'*Res) / (length(Y)-rank(X));

% here we have a problem - the regression is significant, yet it shouldn't
% because the trial to trial variance is the same
figure; subplot(1,2,1); scatter(Q(:,1),Y(:,1),50,'k'); hold on
plot(Q(:,1),Yhat(:,1),'r','LineWidth',3); grid on
t = Beta(1,1) / sqrt(var(1)*([1 0]*inv(X'*X)*[1 0]'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(1,1),t,p);
title(mytitle,'FontSize',14);

subplot(1,2,2); scatter(Q(:,1),Y(:,2),50,'k'); hold on
plot(Q(:,1),Yhat(:,2),'r','LineWidth',3); grid on
t = Beta(1,2) / sqrt(var(2)*([1 0]*inv(X'*X)*[1 0]'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(1,1),t,p);
title(mytitle,'FontSize',14);

%%% _let's now recompute with the conditions_
% the design matrix for conditions is
Xcond = kron(eye(3),ones(50,1));
X = [Xcond Q(:,1) ones(150,1)]; 
figure; tmp = zscore(X(:,4)); imagesc([X(:,[1:3]) tmp/max(tmp) X(:,end)]); 
colormap('gray'); title('Design with condition and beta series','FontSize',14);
Beta = pinv(X)*Y;
Yhat = X*Beta;
Res = Y-Yhat;
var = diag(Res'*Res) / (length(Y)-rank(X));
C = [0 0 0 1 0];

% now that's fine, the counfound variable have been removed
figure; subplot(1,2,1); scatter(Q(:,1),Y(:,1),50,'k'); hold on
plot(Q(:,1),Yhat(:,1),'r','LineWidth',3); grid on
t = C*Beta(:,1) / sqrt(var(1)*(C*pinv(X'*X)*C'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(4,1),t,p);
title(mytitle,'FontSize',14);

subplot(1,2,2); scatter(Q(:,1),Y(:,2),50,'k'); hold on
plot(Q(:,1),Yhat(:,2),'r','LineWidth',3); grid on
t = C*Beta(:,2) / sqrt(var(2)*(C*pinv(X'*X)*C'));
p= 1-spm_Tcdf(t, (size(Y,1)-rank(X)));
mytitle = sprintf('ROI1 to 2 beta=%g \n t=%g p=%g',Beta(4,2),t,p);
title(mytitle,'FontSize',14);


