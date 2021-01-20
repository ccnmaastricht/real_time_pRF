function [X,Y] = Cort2VF(X,Y,varargin)
% transform cortical coordinates to visual field coordinates using the
% inverse of the complex-logarithm transformation described in Schwartz
% (1980) - doi:10.1016/0042-6989(80)90090-5
%
% The function takes as input coordinates on the cortex (with X and Y axes
% parallel and perpendicular to the horizontal meridian, respectively) 
% and returns coordinates in the visual field
% 
% optional input: "Shear" controls the shear parameter. Standard value = .9
% (V1)
%
% Parameter values are taken from:
% J.R. Polimeni, O.P. Hinds, M. Balasubramanian, A. van der Kouwe, 
% L.L. Wald, A.M. Dale, B. Fischl, E.L. Schwartz
% The human V1,V2,V3 visuotopic map complex measured via fMRI at 3 and 7 Tesla
%
%% Handle input
p = inputParser;

defaultAlpha = .9; % V1 shear parameter
addRequired(p,'X',@isnumeric);
addRequired(p,'Y',@isnumeric);
addOptional(p,'Shear',defaultAlpha);

p.parse(X,Y,varargin{:});

X = p.Results.X;
Y = p.Results.Y;
alpha = p.Results.Shear;

%% Parameters
k = 15.0;
a = 0.7;
b = 80;

%% Transformation
W = (X+Y*1i);

t = exp((W+k*log(a/b))/k);
Z = (a-b*t)./(t-1);
Z = abs(Z).*exp(1i./alpha*angle(Z));

X = real(Z);
Y = imag(Z);