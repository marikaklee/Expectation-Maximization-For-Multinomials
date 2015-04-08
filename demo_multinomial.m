%
% Marika Lee
% CS542 A5
% Due 4/7/15
% References in em_multinomial.m
%

function demo_multinomial()
% demo: EM for mixture of multinomials
%
% provided for CS542 A5, Mar 24 2015
% author: Kun He

X = dlmread('ShakespeareMiddleton.txt');
X = X';
[N, M] = size(X);



%%%% PLEASE IMPLEMENT %%%%
%
% Part 1
% 1. run EM with K=2, 10 times, find best solution
% 2. for best solution: show the document IDs assigned to each cluster
% 3. for best solution: plot training log-likelihood vs. iter 
%
k = 2;
for i = 1:10
     [graph, logLikelihood, members, mix, logmu] = em_multinomial(X, k); 
     maxLL=max(logLikelihood);

end

 %display(members');
 display(maxLL);
 display(graph);



%%%% PLEASE IMPLEMENT %%%%
%
% Part 2
% run CV to find best K from {1, 2, 3, 4}
%

% use cross-validation to determine the best K from {1, 2, 3, 4}, 
% using log-likelihood as the selection criterion. 
% When splitting the documents into training and testing folds, 
% only use 2 documents of Shakespeare and 2 from Middleton for 
% testing, since the dataset is small.

testing = [X(:,1:2), X(:,9:10)];
for k = 1:4
    display(k);
    [graph, logLikelihood, members, mix, logmu] = em_multinomial(testing, k); 
    display(logLikelihood);
end



