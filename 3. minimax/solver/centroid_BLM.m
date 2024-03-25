%%************************************************************************
%% Call BLM to solve minimax problems 
function [Y, disthist] = centroid_BLM(X_star, params, options)

A = params.A; 
b = params.b;
mu = params.mu;

N = params.N;
S = params.S;

%% initialization
r = b/(S*(S+1));            % the radius of safety ball
p = (b/(S+1)).*ones(S, N);  % the center of safety ball
X = 0.5*p;

nIter = 20000;
if isfield(options, 'BLM_max_iters')
    nIter = options.BLM_max_iters;
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
    fprintf('\n-------------- BLM ---------------\n');
    fprintf('iter |   err  |   time\n');
end

dist = 1; 
disthist = [dist]; 

%% main loop
for iter = 1:nIter
    
    delta = min(r, 1/iter^(1/3));
    gamma = 1/(6*mu*iter);
    
    % perturbation direction      
    Z = randn(S, N); 
    Z = Z./sqrt(sum(Z.^2, 1));
    
    % query direction
    W = Z - (1/r)*(X - p);
    
    % choose action
    X_hat = X + delta*W; 
    
    % get payoff
    rho = mu*norm(X_hat(:,1))*norm(X_hat(:,1))+X_hat(:,1)'*A*X_hat(:,2)-mu*norm(X_hat(:,2))*norm(X_hat(:,2)); 
    u_hat = [-rho; rho]; 
    
    % estimate gradient
    v_hat = (S/delta)*u_hat'.*Z;
    
    % update pivot
    X_tmp = X + gamma*v_hat;
    X = proj_BLM(X_tmp, b); 
    
    if iter == 1 || mod(iter, checkfreq) == 0
        dist  = norm(X_hat - X_star)/(1 + norm(X_star)); 
    end
    
    if savedisthist == 1, disthist = [disthist; dist]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0))
        fprintf('%5.0f|%0.3e|%3.2e\n', iter, dist, etime(clock, tstart));
    end
end

Y = X_hat;
end