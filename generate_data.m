%% parameters
radius_y = 20;                % radius of cortex in y direction (mm)
radius_x = 55;                % radius of cortex in x direction (mm)
s_rate = 0.5;                 % sampling rate on cortex (mm)

slope = 0.21;                 % slope m in: sigma = m * eccentricity

bound = 0.5;                  % lower bound of sigma

fwhm = 3.5;                   % FWHM of spatial signal point spread (3T = 3.5; 7T = 2)
tau = 2.25;                   % time constant of noise autocorrelation (3T = 2.25; 7T = 1);
var_noise = 0.5;              % noise variance (signa variance = 1)

max_radius = 10;              % maximum visual field radius we measured (in reality this depends on the screen, here we can choose)
resolution = 150;             % resolution of stimulus along 1 axis

num_time = 304;               % number of time points
TR = 2;                       % sampling rate
%% create cortical surface & compute corresponding receptive field parameters
% there is a tight relationship between cortical position and receptive
% field parameters. In the model, we first define the cortical grid and
% then compute the corresponding visual field coordinates.
% Alorithm based on Schwartz et al. (1980)

% subsequently we can convert visual field cartesian coordinates to polar
% coordinates and compute sigma: as a function of eccentricity.
% Parameters based on Freeman & Simoncelli (2011)
x = 0:s_rate:radius_x;
y = -radius_y:s_rate:radius_y;
x_cortex = ones(numel(y), 1) * x;
y_cortex = y' * ones(1, numel(x));

[x_vf, y_vf] = Cort2VF(x_cortex, y_cortex);
mask = x_vf>0; mask = [mask(:,end:-1:1), mask];
x_cortex = [x_cortex(:,end:-1:1), -x_cortex];
y_cortex = [y_cortex(:,end:-1:1), y_cortex];
y_vf = [y_vf(:,end:-1:1), y_vf];
x_vf = [x_vf(:,end:-1:1), -x_vf];

x_cortex = x_cortex(mask);
y_cortex = y_cortex(mask);
x_vf = x_vf(mask);
y_vf = y_vf(mask);

polar_angle = angle(x_vf + y_vf * 1i);
eccentricity = abs(x_vf + y_vf * 1i);
sigma = max(slope * eccentricity, bound);

%% compute point spread kernel
% mimic signal point spread on cortical sheet in visual cortex.
% Parameters based on Shmuel et al 2007
% we later use this to smooth signals and noise across voxels
d_squared = (x_cortex' - x_cortex).^2 + (y_cortex' - y_cortex).^2;
spread = fwhm / (2 * sqrt(2*log(2)));
K = exp(-d_squared ./ (2 * spread^2));

%% compute receptive fields
f = @(mu_x, mu_y, sigma, X, Y) ...
    exp( - ((X - mu_x).^2 + (Y - mu_y).^2) ./ (2 * sigma^2));

r = linspace(-max_radius, max_radius, resolution);
[X, Y] = meshgrid(r);
X = X(:);
Y = -Y(:);

num_pixels = resolution^2;
num_voxels = numel(x_vf);
W = zeros(num_voxels, num_pixels);

for v=1:num_voxels
    W(v, :) = f(x_vf(v), y_vf(v), sigma(v), X, Y);
    W(v, :) = W(v,:) / sum(W(v, :));
end

%% create noise for all voxels
% create noise as Wiener process with tau based on estimates of real data
dt = 1e-1;
t_sim = num_time * TR;
t_steps = floor(t_sim / dt);
dsig = sqrt(dt / tau);
noise = zeros(t_steps, num_voxels);

for t=2:t_steps
    noise(t,:) = noise(t-1,:) - dt * noise(t-1,:) / tau + dsig * randn(1, num_voxels);
end

noise = noise(1:TR/dt:end,:);
noise = zscore(noise * K) * var_noise;

%% create simulated data
% create data based on bar stimulus and an empirically measured V1 HRF and
% add noise
load('stimulus')
two_gamma = @(t) (6*t.^5.*exp(-t))./gamma(6)...
    -1/6*(16*t.^15.*exp(-t))/gamma(16);
t = (0:TR:34)';
hrf = two_gamma(t);
len_hrf = numel(hrf);
hrf = [hrf; zeros(num_time-len_hrf,1)];
hrf_fft = fft(hrf);
Y = (W * stimulus)';
Y_fft = fft(Y);

Y = ifft(Y_fft .* hrf_fft);
Y = Y * K;
Y = (Y - mean(Y)) / max(std(Y)) + noise;

%% Post-processing
data = Y;
idx = [1:num_voxels];
idx = idx(eccentricity<=max_radius); % remove voxels lying beyond visual field
data = zscore(Y(:,idx));
W = W(idx,:);
x_vf = x_vf(idx);
y_vf = y_vf(idx);
sigma = sigma(idx);
mn = min(W,[],2)
mx = max(W,[],2);
W = ((W - mn) ./ (mx - mn)); %Normalize weights (RFs) between [0,1]
