function [f, g, H] = subprob_LZBZ(p, X, v, gamma, mu, t, d)  

N = length(X); 

% Calculate objective f
f = gamma*v'*(X-p) + (gamma*(t+1)*mu)*norm(X-p)^2 - sum(log(d+p)) - sum(log(d-p)) ...
    + sum(log(d+X)) + sum(log(d-X)) - (1./(d-X) - 1./(d+X))'*(p-X);

if nargout > 1 % gradient required
    g = -gamma*v + 2*gamma*(t+1)*mu*(p-X) - (1./(d+p) - 1./(d-p) - 1./(d+X) + 1./(d-X));
end

if nargout > 2 % Hessian required
    H = 2*gamma*(t+1)*mu*eye(N) + diag(1./((d+p).^2)) + diag(1./((d-p).^2)); 
end

end
