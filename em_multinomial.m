% Marika Lee
% CS542 A5
% Due 4/7/15
%
% References: 
%
% Expectation/Maximization Algorithm (for Multinomials): 
% <used for format of helper methods needed for em_multinomial.m>
% http://www.montefiore.ulg.ac.be/~dteney/DML/Statistics/
% General format for EM Algorithm (for Gaussian), much like demo_gaussian.m: 
% <review of how a gaussian EM algorithm works>
% http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model
% Definition/Pseudocode for generalized EM: 
% <used as pseudocode of how to format em multinomial>
% http://csg.sph.umich.edu/abecasis/class/2006/615.18.pdf
%

function [graph, loglikelihoods, members, mix, logmu] = em_multinomial(data, K)
% added graph in output to use in demo_multinomial.m

% EM for mixture of multinomials
%   input: 
%                X - training data (rows are vectors) - in demo_multinomial.m
%                K - # of clusters 
%   output: 
%            graph - log likelihood vs iteration plot
%   loglikelihoods - training log-likelihoods, all iterations
%          members - cell array, document IDs in each cluster
%              mix - mixing coefficients (\pi)
%            logmu - log of multinomial parameters (\mu)
    
%
% provided for CS542 A5, Mar 24 2015
% author: Kun He

% read data
data = dlmread('ShakespeareMiddleton.txt');
data = data';

%%%% PLEASE IMPLEMENT %%%%
%
% 0. randomly initialize
%
% note: initialize responsibility values tau instead of multinomial parameters
% and start by an M-step. Reason: the search space is very large and we could get 
% very far from a good solution if we directly initialize multinomial parameters 


%N=18 columns
%M=10025 row
R = initialization(data, K);
 
loglikelihoods = [];
converged = false;
threshold = 1e-6;
prev = -inf;
iter = 1;

% 1. EM iterations
max_iters = 100;
logL = -inf(1, max_iters);

while ~converged & (iter < max_iters)
  iter = iter + 1;
  
  	%%%% PLEASE IMPLEMENT %%%%
	%
	% 1.1 M-step
	%
    
  [mix, logmu, sigma] = maximization(data, R);

  	%%%% PLEASE IMPLEMENT %%%%
	%
	% 1.2 E-step
	%
  [R, logL(iter)] = expectation(data, mix, logmu, sigma);
  
    
  converged = logL(iter) - logL(iter - 1) < threshold * abs(logL(iter));  
  %fprintf('EM iter%02d: log-likelihood = %g\n', (iter-1), logL(iter));

end


  	%%%% PLEASE IMPLEMENT %%%%
	%
	% 1.3 compute current log-likelihood (logL)
	%
[logL] = logL(2:iter);
 
if ~converged
  printf('Clustering did not converge in %d steps.\n', max_iters);
end


% 1.4 convergence check
if (logL < prev)
    % note: Theoretically the likelihood never decreases during EM. In practice 
    % small decrements are possible in numerical computation, especially near 
    % convergence (if not small, then there's a bug)
    fprintf('Warning: decrease of %g in log-likelihood.\n', logL-prev);
end
if (abs(logL-prev) < 1e-6)
    converged = true;
    fprintf('Converged.\n');
end
prev = logL;
    
loglikelihoods = max(logL);


%%%% PLEASE IMPLEMENT %%%%
%
% 2. final cluster membership
%

% [~, members(1, :)] = max(R, [], K);
% %display(members');

display_members = members(1:18)';
%display(display_members);
cluster = cell(1, K);
for i = 1:18
    if (display_members(i)==1)
        %%display(i);
    end
end
%display(cluster{1});


% Graphing
clf;
graph = plot((1:(iter-1)),logL);
xlabel('Iteration');
ylabel('Log-Likelihood');
title(loglikelihoods);

 
%-------------------------------MAXIMIZATION----------------------------
function [mix, logmu, sigma] = maximization(data, R)
    
    [d n] = size(data); %d=18, n=10025
    %%display(size(data));
    K = size(R, 2);
    
    regularization = eye(d) * threshold; %for covariance

    s = sum(R, 1);
    mix = s / n;

    logmu = bsxfun(@rdivide, data * R, s);
    sigma = zeros(d, d, K);

    for i = 1:K
      Xo = bsxfun(@minus, data, logmu(:, i));
      Xo = bsxfun(@times, Xo, sqrt(R(:, i)'));
      sigma(:, :, i) = (Xo * Xo' + regularization) / s(i);
    end
    
end
 
%-------------------------------EXPECTATION-----------------------------
function [R logL] = expectation(data, mix, logmu, sigma)
% compute the responsibility values {?i,k}
% To avoid numerical underflow and divide-by-zero problems, 
% you should work with log probabilities: li,k = log(p(zi = k, xi)). 
% A handy trick is to find Ai = maxKk=1 lk,i, then compute ?i,k 

  n = size(data, 2); %n: 10025
  K = size(logmu, 2); %K: 2
  
  R = zeros(n, K);
 

  for i = 1:K
      R(:, i) = log_gauss_pdf(data, logmu(:, i), sigma(:, :, i));

  end
  R = bsxfun(@plus, R, log(mix));

  T = log_sum_exp(R, 2);
  logL = sum(T) / n; % Log likelihood
  
 
  R = bsxfun(@minus, R, T);
  R = exp(R);
end

%------------------------INITIALIZATION---------------------------
function R = initialization(data, K)
% search space is very large, so if we?re not careful we will likely start 
% far away from a good solution. One way to handle it is to initialize the 
% responsibility values {?i,k} rather than directly initializing the 
% distribution parameters (?,?), then start by running the M-step.
%     
    n = size(data, 2); %n: 10025
    %n=18;
    
    if K == 1
        members = ones(1, n);
    end
    
    if K > 1
      
        % Random initialization
        idx = randsample(n, K); %2x1
        %%display(size(idx));

        m = data(:,idx);

        %m=[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]';
        %m = [1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]';
        % m = [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0; 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]';
        %%display(size(m)); %18x2
        %display(m);

        %18x10025 * 10025x2
        % subtract
        minus1 = m' * data; %2x18 * 18x10025 = 2x10025
        %%display(size(minus1));
        %%display(minus1);
        minus2 = sum(m.^2, 1)' / 2; %2x1
        %%display(size(minus2));

        %2x10025 - 2x1 = 2x10025
        [~, members] = max(bsxfun(@minus, minus1, minus2));
        %%display(members);
    end
    

    R = full(sparse(1:n, members, 1, n, K, n));
    %%display(members);
    %R =[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1;1  1 1 1 1 1 1 1 0 0 0 0 0 0 0 0];


    %%display(R);
end
 

%----------------------------Log Gaussian PDF-------------------------
function y = log_gauss_pdf(data, logmu, sigma)
  d = size(data, 1);
  [R p]= cholcov(sigma, 0);
  data = bsxfun(@minus, data, logmu);
  y = -((d * log(2 * pi) + 2 * sum(log(diag(R)))) + (sum((R' \ data).^2, 1))) / 2;
end
 
%--------------------------------Log Sum Exp-----------------------------
function s = log_sum_exp(x, dim)

  y = max(x, [], dim);
  x = bsxfun(@minus, x, y);
  s = y + log(sum(exp(x), dim));
  
  
  
  
  i = find(~isfinite(y));
  
  if ~isempty(i)
      s(i) = y(i);
  end
  
  
  
end
 
end

%----------------------------------eof-------------------------------

