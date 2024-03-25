function [func, grad] = logit(X, A, b, mu)  

[m, ~] = size(A); 

% Calculate objective f
z = A*X; 
[p1, p2, p3] = phi(z); 
func = - (b'*p2 + (1-b)'*p3) / m; 
func = func + mu * norm(X) * norm(X); 
 
if nargout > 1 % gradient required
    grad = A' * (p1 - b) / m;
    grad = grad + 2 * mu * X; 
end

end

%% auxillary function
function [p1, p2, p3] = phi(X)

n = length(X); 
idx = find(X > 0);
nidx = setdiff(1:n, idx); 

% Sigmoid: 1 / (1 + exp(-t))
p1 = zeros(n, 1);
p1(idx) = 1./(1 + exp(-X(idx))); 
exp_X = exp(X(nidx));
p1(nidx) = exp_X./(1 + exp_X); 

if nargout > 1 % log(Sigmoid): log(1 / (1 + exp(-t)))
    p2 = zeros(n, 1);
    p2(idx) = -log(1 + exp(-X(idx))); 
    p2(nidx) = X(nidx) - log(1 + exp(X(nidx))); 
end

if nargout > 2 % log(1-Sigmoid): log(1 - 1 / (1 + exp(-t)))
    p3 = zeros(n, 1);
    p3(idx) = - X(idx) - log(1 + exp(-X(idx))); 
    p3(nidx) = -log(1 + exp(X(nidx))); 
end

end
