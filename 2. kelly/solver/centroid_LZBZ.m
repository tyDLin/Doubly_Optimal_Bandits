%%************************************************************************
%% Call LZBZ to solve Kelly auction
function [Y, disthist] = centroid_LZBZ(X_star, params, options)

b = params.b;
g = params.g;
d = params.d;
q = params.q;

N = params.N;
S = params.S;

%% initialization
X = 0.1*(b/(S+1)).*ones(S, N);

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
 
% dist = norm((b/(S+1)).*ones(S, N) - X_star)/(1+norm(X_star)); 
dist = 5; 
disthist = [dist]; 

%% main loop
for iter = 1:nIter
        
    gamma = 1/(2*N*(iter)^(1/2));

    % perturbation direction  
    Z = randn(S, N); 
    Z = Z./sqrt(sum(Z.^2, 1));
    
    % scaling matrix
    A1 = diag(1./(b - sum(X, 1)).^2 + gamma*(iter+1)*g'); 
    A2 = kron(A1, eye(S)) + diag(1./reshape(X.^2, S*N, 1)); 
    A = sqrtm(A2); 

    % choose action
    X_hat = X + reshape(inv(A)*reshape(Z, S*N, 1), S, N); 

    % get payoff
    rho = (q./(sum(X_hat, 2) + d)).*X_hat;
    u_hat = g.*sum(rho, 1)' - sum(X_hat(:)); 
        
    % estimate gradient
    v_hat = S*u_hat'.*reshape(A*reshape(Z, S*N, 1), S, N);
    
    % update pivot 
    tmp = proj_LZBZ(X, v_hat, gamma, g, iter, b); 
    X = tmp;

    if iter == 1 || mod(iter, checkfreq) == 0
        dist  = norm(X_hat - X_star)/(1+norm(X_star)); 
    end
    
    if savedisthist == 1, disthist = [disthist; dist]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0))
        fprintf('%5.0f|%0.3e|%3.2e\n', iter, dist, etime(clock, tstart));
    end
end

Y = X_hat;
end