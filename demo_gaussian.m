function demo_gaussian(testcase)
% demonstration of EM for 2D Gaussian mixture models
%
% provided for CS542 A5, Mar 24 2015
% author: Kun He
% modified from Tony Jebara's implementation 

if nargin < 1, testcase = 1; end

% load data
if testcase == 1
	X = dlmread('ShakespeareMiddleton.txt');
else
	X = dlmread('gaussB.txt');
end
N = size(X, 1);
D = size(X, 2);
K = 3;  % number of Gaussians to fit

% initialize
[mu, covar, mix] = rand_init(X, K);
loglikelihoods = [];
converged = false;
prev = -inf;
iter = 0;

% EM iterations
figure; clf
max_iters = 1000;
while ~converged & (iter < max_iters)
	iter = iter + 1;

	% E-step
	tau = zeros(N, K);
	for k = 1:K
		detcov = det(covar{k});
		invcov = (covar{k} + 1e-6*eye(D)) \ eye(D);  % add epsilon to diagonal to ensure PSDness
		dev = X - repmat(mu(k, :), [N 1]);
		tau(:,k) = mix(k)*(2*pi)^(-D/2)/sqrt(detcov)*diag(exp(-.5*dev*invcov*dev'));
	end

	% compute current log-likelihood
	L = zeros(1, N);
	for n = 1:N
		l = sum(tau(n, :));
		L(n) = l;
		tau(n, :) = tau(n, :)/l;
	end
	logL = sum(log(L));
	loglikelihoods = [loglikelihoods, logL];
	
    fprintf('EM iter%02d: log-likelihood = %g\n', iter, logL);

	% convergence check
	if (logL < prev)
		% note: Theoretically the likelihood never decreases during EM. In practice 
		% small decrements are possible in numerical computation, especially near 
		% convergence (if not small, then there's a bug)
		fprintf('Warning: decrease of %g in log-likelihood.\n', logL-prev);
	end
	if (logL-prev < 1e-6)
		converged = true;
		fprintf('Converged.\n');
	end
	prev = logL;

	% M-step
	for k = 1:K
		sumtau = sum(tau(:, k));
		% mixing weight
		mix(k) = sumtau/N;
		% mean 
		mu(k, :) = sum(diag(tau(:, k)) * X, 1) / sumtau;
		% convariance
		covar{k} = zeros(D);
		for n = 1:N
			dev = X(n, :) - mu(k, :);
			covar{k} = covar{k} + tau(n,k)*dev'*dev;
		end
		covar{k} = covar{k}/sumtau;
	end

	% visualize
	subplot(3, 1, 1:2)
	plot(X(:,1),X(:,2),'g.'); hold on;
	plotClust(mu,covar,1,2); hold off;
	subplot(3, 1, 3);
	plot(loglikelihoods);
	drawnow;
end

% ----------------------------------------------------------------------

function [mu, covar, mix] = rand_init(X, K)
%
% function [mu,covar,mix] = rand_init(X,K)
%
%  X     the data, columns are dimensions, rows are points
%  K     number of mixtures
%
%  mu    matrix, means (cols are dimensions)
%  cov   cell array, covariance matricies
%  mix   vector, mixing weights

vscale = 2;

[N, D] = size(X); 	

% Randomly initialize K Gaussians
mx = max(X);
mn = min(X);
initmu = (mx+mn)/2;
initsd = (mx-mn)/(K^(1/D));
initcv = diag((initsd/vscale).^2, 0);

mix = ones(K, 1)/K;
mu = [];
covar = cell(1, K);

for i = 1:K
	mu = [mu; (rand(size(mx)).*(mx-mn) + mn)];

	q = rand(size(initcv)) - 0.5;
	q = 2*q*q'*mean(initsd);
	scal = (mean(initsd)/K)^2;
	q = scal * eye(size(initcv));
	covar{i} = q;
end

% ----------------------------------------------------------------------

function plotClust(mu,c,d1,d2)
% plotClust(mu,c,d1,d2)
%
% PLOT THE PROJECTION OF A GAUSSIAN CLUSTER
% This function plots the projection of a gaussian cluster in two dimensions.

[K, D] = size(mu);
for i = 1:K
	plotGauss(mu(i,d1), mu(i,d2), c{i}(d1,d1), c{i}(d2,d2), c{i}(d1,d2));
end

% ----------------------------------------------------------------------

function plotGauss(mu1,mu2,var1,var2,covar)
% plotGauss(mu1,mu2,var1,var2,covar)
%
% PLOT A 2D Gaussian
% This function plots the given 2D gaussian on the current plot.

t = -pi:.01:pi;
k = length(t);
x = sin(t);
y = cos(t);

R = [var1 covar; covar var2];

[vv,dd] = eig(R);
A = real((vv*sqrt(dd))');
z = [x' y']*A;

plot(mu1,mu2,'X'); hold all;
plot(z(:,1)+mu1,z(:,2)+mu2,'.');

