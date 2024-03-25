function [f, g, H] = subprob_LZBZ(p, X, v, gamma, b, t, d)  

N = length(X); 

% Calculate objective f
f = gamma*v'*(X-p) + (gamma*(t+1)*b/2)*norm(X-p)^2 - sum(log(p)) - sum(log(d-p)) ...
    + sum(log(X)) + sum(log(d-X)) - (1./(d-X) - 1./X)'*(p-X);

if nargout > 1 % gradient required
    g = -gamma*v + gamma*(t+1)*b*(p-X) - (1./p - 1./(d-p) - 1./X + 1./(d-X));
end

if nargout > 2 % Hessian required
    H = gamma*(t+1)*b*eye(N) + diag(1./(p.^2)) + diag(1./((d-p).^2)); 
end

end
