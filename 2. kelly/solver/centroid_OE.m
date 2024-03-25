%%************************************************************************
%% Call OE to solve NE for Kelly auction
function Y = centroid_OE(params, options)

b = params.b;
g = params.g;
d = params.d;
q = params.q;

N = params.N;
S = params.S;

%% initialization

X0 = 0.1*(b/(S+1)).*ones(S, N); % initial point
X1 = X0;

nIter = 2000;
lam = 1;
gam = 0.01;

if isfield(options, 'nIter')
    nIter = options.nIter;
end

if isfield(options, 'lam'),       lam = options.lam;            end    
if isfield(options, 'gam'),       gam = options.gam;            end    

tstart = clock;
display = 1;         % option of displaying
displayfreq = 100;   % gap of display
checkfreq = 1;       % frequency of check
savedisthist = 0;    % save distance history
 
if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end   
if isfield(options, 'savedisthist'),  savedisthist = options.savedisthist;  end
 
if display == 1
    fprintf('\n-------------- OE ---------------\n');
    fprintf('iter |   err  |   time\n');
end
 
errhist = []; 

%% main loop
for iter = 1:nIter
    
    % calculate gradient at X0, X1      
    total_share0 = sum(X0, 2) + d;     % total share for each resource
    F0 = - (q./total_share0.*ones(S, N) - (q./(total_share0).^2).*X0).*g' + ones(S, N);
    total_share1 = sum(X1, 2) + d;     % total share for each resource
    F1 = - (q./total_share1.*ones(S, N) - (q./(total_share1).^2).*X1).*g' + ones(S, N);
    
    % compute X(t+1)    
    obj = @(X)subprob_OE(X, X1, F0, F1, gam, lam);

    % create contraint matrix A
    A = zeros(N, S*N);
    for i = 1:N
        A(i, (1+(i-1)*S):(i*S)) = ones(1, S);
    end
    options1 = optimoptions(@fmincon, 'Display', 'off');

    tmp = fmincon(obj, reshape(X1,N*S,1), A, b*ones(N,1), [], [], zeros(N*S,1), [], [], options1);

    % update 
    X0 = X1;
    X1 = reshape(tmp, S, N);
    
    if iter == 1 || mod(iter, checkfreq) == 0
        err  = norm(X1 - X0)/(1 + norm(X0) + norm(X1)); 
    end
     
    if savedisthist == 1, errhist = [errhist; err]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0))
        fprintf('%5.0f|%0.3e|%3.2e\n', iter, err, etime(clock, tstart));
    end
end

Y = X1; 
end