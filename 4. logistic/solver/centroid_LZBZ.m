%%************************************************************************
%% Call BLM to solve regularized logistic regression
function [Y, disthist] = centroid_LZBZ(X_star, params, options)

A = params.A;
b = params.b;
mu = params.mu;

[~, N] = size(A);
d = params.d*ones(N, 1); 

%% initialization
X = zeros(N, 1);
L = 0.25*max(sum(A.*A, 2));

nIter = 20000;
if isfield(options, 'LZBZ_max_iters')
    nIter = options.LZBZ_max_iters;
end

tstart = clock;
display = 1;         % option of displaying
displayfreq = 1;     % gap of display
checkfreq = 1;       % frequency of check
savedisthist = 0;    % save distance history

if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end   
if isfield(options, 'savedisthist'),  savedisthist = options.savedisthist;  end

if display == 1
    fprintf('\n-------------- LZBZ---------------\n');
    fprintf('iter |   err  |   time\n');
end
 
dist = 1; 
disthist = [dist]; 

%% main loop
for iter = 1:nIter
    
    gamma = min(1/(2*L), 1/(2*N*(iter)^(1/2)));
 
    % perturbation direction  
    Z = randi(2,N,1)*2-3; 

    % scaling matrix
    S = (1./((d+X).^2) + 1./((d-X).^2) + 2*gamma*mu*(iter+1)).^(-1/2);

    % choose action
    X_hat = X + S.*Z; 

    % get payoff
    u_hat = - logit(X_hat, A, b, mu);
    
    % estimate gradient
    v_hat = (u_hat./S).*Z;
    
    % update pivot
    tmp = proj_LZBZ(X, v_hat, gamma, mu, iter, d); 
    X = tmp;

    if iter == 1 || mod(iter, checkfreq) == 0
        dist = norm(X_hat - X_star)/(1 + norm(X_star)); 
    end
    
    if savedisthist == 1, disthist = [disthist; dist]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0))
        fprintf('%5.0f|%0.3e|%3.2e\n', iter, dist, etime(clock, tstart));
    end
end

Y = X_hat;
end