%%************************************************************************
%% Call AGD to solve NE for regularized logistic regression
function Z = centroid_AGD(params, options)

A = params.A;
b = params.b;
mu = params.mu;
d = params.d; 

[~, N] = size(A);

%% initialization
X = zeros(N, 1); % initial point
Y = X; 

nIter = 20000;
L = 0.25*max(sum(A.*A, 2));
kappa = L/(2*mu);

if isfield(options, 'nIter')
    nIter = options.nIter;
end  

tstart = clock;
display = 1;         % option of displaying
displayfreq = 100;   % gap of display
checkfreq = 1;       % frequency of check
savedisthist = 0;    % save distance history
savetimehist = 0;    % save time history
 
if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end   
if isfield(options, 'savedisthist'),  savedisthist = options.savedisthist;  end
if isfield(options, 'savetimehist'),  savetimehist = options.savetimehist;  end
 
if display == 1
    fprintf('\n-------------- AGD ---------------\n');
    fprintf('iter |   err  |   time\n');
end
 
errhist = []; 

%% main loop
for iter = 1:nIter
    
    % perform gradient step at X   
    [~, g] = logit(X, A, b, mu); 
    Y_pre = Y; 
    Y = max(-d, min(d, X - g/L)); 
    
    % compute the next iterate
    X = Y + ((sqrt(kappa)-1)/(sqrt(kappa)+1))* (Y - Y_pre);  
    
    if iter == 1 || mod(iter, checkfreq) == 0
        err  = norm(Y - Y_pre)/(1 + norm(Y) + norm(Y_pre)); 
    end
     
    if savedisthist == 1, errhist = [errhist; err]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0))
        fprintf('%5.0f|%0.3e|%3.2e\n', iter, err, etime(clock, tstart));
    end
end

Z = Y; 
end