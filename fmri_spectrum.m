function fmri_spectrum

% simple routine to look at the spectrum of fMRI time series
% data in are expected to be a phantom as to have signal somewhere
% simply read the signal in time from the phantom and outside it
% plot the frequency spectrum for each given the TR
%
% needs SPM (http://www.fil.ion.ucl.ac.uk/spm/) installed in the path 
% Cyril Pernet 12 Januray 2016 - University of Edinburgh
% -----------------------------------------------------

P = spm_select(Inf,'image','select fmri images');
V = spm_vol(P);
data = spm_read_vols(V); % 4D matrix
TR = input('TR in sec? ');
sampling_rate = 1/TR; % Hz

% create mean image
average = mean(data,4);
A = V(1); [root,~,ext]=fileparts(A.fname);
cd(root); A.fname = [pwd filesep 'average' ext];
A.descrip = 'mean of all images';
spm_write_vol(A,average);

% find signal 
[~,voxel_value]=hist(average(:),3); % only a phantom so 3 bins is enough
dist =  diff(voxel_value); dist = dist(1); 
threshold = voxel_value(3) - dist/2;
[L,NUM] = bwlabeln(average>threshold); % find the phantom
if NUM ~= 1
    error('cannot locate the phantom in the average image - program stopped')
end

% freq analysis
[x,y,z] = ind2sub(size(average),find(L)); % look at the phantom
time_series = spm_get_data(V,[x y z]');
signal_freq = freq_spec(time_series,sampling_rate);
N = size(time_series,2);

[x,y,z] = ind2sub(size(average),find(L==0)); % look outside the phantom
time_series = spm_get_data(V,[x y z]');
rand_select = randperm(size(time_series,2));
rand_select = rand_select(1:N); % take the same number of voxel as the phantom, but randomly
noise_freq = freq_spec(time_series(:,rand_select),sampling_rate);

% finally do a figure
% normalize by the number of time points
Nyquist_Limit = (1/sampling_rate)/2; 
nx = size(time_series,1)/2;
x = linspace(0,1,nx)*Nyquist_Limit;
y = mean(signal_freq(1:nx,:),2);
figure; plot(x,y,'r','LineWidth',3); 
yy = mean(noise_freq(1:nx,:),2);
hold on; plot(x,yy,'LineWidth',3);
xlabel('frequency','Fontsize',12); 
ylabel('amplitude','Fontsize',12)
axis([-0.0001 0.005 1  max(max([y yy]))]); grid on; 
legend('phantom','empty space')



end

function coef = freq_spec(time_series,sampling_rate)

time = (0:sampling_rate:1)';  
for t=1:size(time_series,2)
coef(:,t) = abs(fft(time_series(:,t)) / length(time)); 
end

end




