% fwhm here is normalized with stimulus dimension and should be in the range [0,1].
% For example for a fwhm of 0.5 is actually 75 (w.r.t 150)
% The size of data matrix should be time_volumes-by-voxels
% The size of the stimulus matrix should be time_volums-by-pixels

shrink = 6;
parameters.fwhm = 0.15;
parameters.eta = 0.1;
parameters.f_sampling = 1/2;															 % 1/TR
parameters.r_stimulus = 150; 															 % stimulus size is 150x150
parameters.n_features = 250; 															 % number of tilings
parameters.n_gaussians = 5;																 % number of guassians within a tile
parameters.n_voxels = size(data,2);
prf_mapper = HGR(parameters);
data = zscore(data);
[mask,corrfit] = prf_mapper.get_best_voxels(data,stimulus);% select top visual voxels

% Estimate pRF paramters
% Note that this procedure will be only performed on masked voxels.
% The pRF parametrs corresponding to the voxels outside of mask will be filled with "NaN"s.
rf_param = prf_mapper.get_parameters('alpha',shrink,'max_radius',10,'mask',mask);

%Get masked estimated pRF parameters

est_x = rf_param.mu_x(mask);
est_y = rf_param.mu_y(mask);
est_sigma = rf_param.sigma(mask);

% Mask out ground-truth pRF parameters

x_vf = x_vf(mask);	 																			 % X-coordinate
y_vf = y_vf(mask);	 																			 % Y-coordinate
sigma = sigma(mask); 																			 % Standard Deviation

% Get population receptive fields for masked voxels

data = data(:,mask);
parameters.n_voxels = size(data,2);
prf_mapper.ridge(data,stimulus);
gam = prf_mapper.get_features;
theta = prf_mapper.get_weights;
Wff = (gam * theta)';
mn = min(Wff,[],2);
mx = max(Wff,[],2);
Wff = ((Wff - mn) ./ (mx - mn));
Wff = Wff.^shrink;
Wff(isnan(Wff)) = 0; 																			% Estimated pRFs
W = W(mask,:); 																						% Ground-truth pRFs
