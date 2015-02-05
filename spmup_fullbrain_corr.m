function spmup_fullbrain_corr

% full brain connectivity based on clustering of skipped Spearman correlations
%
% FORMAT spmup_fullbrain_corr
%
% OUTPUT: A map of clustered regions is saved on the drive
%
% this appoach has several advantages:
% 1 - using correlations has the advantage of getting rid of differences in local
% activation level
% 2 - using a skipped Spearman correlation allows for monotonic not just
% linear correlations while also getting rid a increased correlations due
% to artefacts
% 3 - the clustering is designed to aggregate pairs without a priori number
% of regions
%
% Known issue: the code make use of matlab nan functions + it is memory
% hungry, by loading all images at once and also making a large N*N matrix
% with all the in mask voxels (ie works with a 4Gb+ machine)
%
% Cyril Pernet July 2014

[t,sts] = spm_select(Inf,'image','select files');
if isempty(sts)
    return
end
V = spm_vol(t);
images = spm_read_vols(V);
mask = spmup_auto_mask(images);

% compute the correlation matrix
indices = find(mask); % which voxels to analyze
[x,y,z] = ind2sub(size(mask),indices); % coordinates of those voxels
N = length(indices); % number of voxels to analyze
Corr_Matrix = nan(N,N); % matrix to store correlations

for v = 1:N
    for o =1:N
        
        % Corr_Matrix(v,o) = corr(squeeze(images(x(v),y(v),z(v),:)),squeeze(images(x(o),y(o),z(o),:)));
        
        % use skipped Spearman correlation
        a = squeeze(images(x(v),y(v),z(v),:));
        b = squeeze(images(x(o),y(o),z(o),:));
        X = [a b]; result=mcdcov(X,'cor',1,'plots',0,'h',floor((n+size(X,2)*2+1)/2));
        center = result.center;
        
        vec=1:n;
        for i=1:n % for each row
            dis=NaN(n,1);
            B = (X(i,:)-center)';
            BB = B.^2;
            bot = sum(BB);
            if bot~=0
                for j=1:n
                    A = (X(j,:)-center)';
                    dis(j)= norm(A'*B/bot.*B);
                end
                % IQR rule
                [ql,qu]=idealf(dis);
                record{i} = (dis > median(dis)+gval.*(qu-ql)) ;
            end
        end
        clear result center dis
        
        try
            flag = nan(n,1);
            flag = sum(cell2mat(record),2); % if any point is flagged
            
        catch ME  % this can happen to have an empty cell so loop
            flag = nan(n,size(record,2));
            index = 1;
            for s=1:size(record,2)
                if ~isempty(record{s})
                    flag(:,index) = record{s};
                    index = index+1;
                end
            end
            flag(:,index:end) = [];
            flag = sum(flag,2);
        end
        keep=vec(~flag);
        clear record vec flag
        
        % Spearman correlation
        xrank = tiedrank(a(keep),0); yrank = tiedrank(b(keep),0);
        Corr_Matrix(v,o) = sum(detrend(xrank,'constant').*detrend(yrank,'constant')) ./ ...
            (sum(detrend(xrank,'constant').^2).*sum(detrend(yrank,'constant').^2)).^(1/2);
        clear xrank a yrank b keep
    end
    
end

% fill the lower part of the correlation matrix
Corr_Matrix = Corr_Matrix + Corr_Matrix.' - eye(size(Corr_Matrix)) .* Corr_Matrix ;

% do the clustering
% median values
pref = nanmedian(Corr_Matrix);
[clusters,net_similarity,similariy_values,~]=apcluster(Corr_matrix,pref,'plot','nonoise');
cluster_values = unique(clusters);

% now put clustering results in brain space with labels
mask(indices) = 0;
for c = 1:size(cluster_values,1)
    A = x(clusters == cluster_values(c));
    B = y(clusters == cluster_values(c));
    C = z(clusters == cluster_values(c));

   % add to a overall brain result
    mask(A,B,C) = c;
end
