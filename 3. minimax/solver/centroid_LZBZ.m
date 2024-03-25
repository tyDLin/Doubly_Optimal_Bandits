%%************************************************************************
%% Call LZBZ to solve minimax problems 
function [Y, disthist] = centroid_LZBZ(X_star, params, options)

A = params.A; 
b = params.b;
mu = params.mu;

N = params.N;
S = params.S;

%% initialization
X = 0.5*(b/(S+1)).*ones(S, N);

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

    gamma = 1/(4*S*(iter)^(1/2));
 
    % perturbation direction  
    Z = randn(S, N); 
    Z = Z./sqrt(sum(Z.^2, 1));
    
    % scaling matrix
    B1 = diag(1./(b - sum(X, 1)).^2 + gamma*(iter+1)*ones(1, N)); 
    B2 = kron(B1, eye(S)) + diag(1./reshape(X.^2, S*N, 1)); 
    B = sqrtm(B2); 

    % choose action
    X_hat = X + reshape(inv(B)*reshape(Z, S*N, 1), S, N); 

    % get payoff
    rho = mu*norm(X_hat(:,1))*norm(X_hat(:,1))+X_hat(:,1)'*A*X_hat(:,2)-mu*norm(X_hat(:,2))*norm(X_hat(:,2)); 
    u_hat = [-rho; rho]; 
    
    % estimate gradient
    v_hat = S*u_hat'.*reshape(B*reshape(Z, S*N, 1), S, N);
    
    % update pivot 
    tmp = proj_LZBZ(X, v_hat, gamma, iter, b); 
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