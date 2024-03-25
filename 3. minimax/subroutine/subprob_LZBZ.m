function [f, grad, Hess] = subprob_LZBZ(p, X, v, gamma, t, b)  

[S, N] = size(X); 
X = reshape(X, S*N, 1); % transform matrix to vector
v = reshape(v, S*N, 1); % transform matrix to vector

% Calculate objective f
R  = -sum(log(X)) - sum(log(b-sum(reshape(X, S, N), 1)));
Rp = -sum(log(p)) - sum(log(b-sum(reshape(p, S, N), 1)));
gR = - 1./X + kron(1./(b-sum(reshape(X, S, N), 1))', ones(S,1));
gRp = - 1./p + kron(1./(b-sum(reshape(p, S, N), 1))', ones(S,1));
f = gamma*v'*(X-p) + 1/2*gamma*(t+1)*norm(X-p)*norm(X-p) + Rp - R - gR'*(p-X);

if nargout > 1 % gradient required
    grad = -gamma*v + gamma*(t+1)*norm(X-p)*norm(X-p) + gRp - gR;
end

if nargout > 2 % Hessian required
    tmp = diag(1./(b - sum(reshape(X, S, N), 1)).^2 + gamma*(t+1)*ones(1, N)); 
    Hess = kron(tmp, eye(S)) + diag(1./X.^2);
end

end
