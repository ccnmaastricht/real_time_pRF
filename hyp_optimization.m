% Sample script for optimizing hyperparameters (learning rate (eta) and fwhm) using bayesian optimization
% The loss function used here is Jaccard Distance between mapped RFs (Wff) and ground-truth RFs (GTWff)

rng default
eta = optimizableVariable('e',[1e-34,10]); 	% set range for the values of eta
fwhm = optimizableVariable('f',[0,1]);     	% set range for the values of fwhm
objfun=@(x)loss_func(x.e,x.sh,data,stimulus,GTWff);
results = bayesopt(objfun,[eta,shrink,fwhm],'Verbose',1,...
    'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',60)

hyp_best = bestPoint(results);							% best hyperparameters


% Define loss function


function loss = loss_func(data,stimulus,GTWff)

	parameters.fwhm = fwhm;
	parameters.eta = eta;
	shrink = 6;
	parameters.f_sampling = 1/2;
	parameters.r_stimulus = 150;
	parameters.n_features = 250;
	parameters.n_gaussians = 5;
	parameters.n_voxels = size(data,2);
	prf_mapper = HGR(parameters);
	data = zscore(data);
	[mask,corrfit] = prf_mapper.get_best_voxels(data,stimulus)
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
	Wff(isnan(Wff)) = 0;
	loss = JD(Wff,GTWff);

end

% Jaccard Distance

function distance = JD(x,y)

	distance = (sum(min(x(:),y(:)))/(sum(max(x(:),y(:))));

end
