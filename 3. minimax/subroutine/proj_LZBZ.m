function Y = proj_LZBZ(X, v, gamma, t, b)  

[S, N] = size(X);

options = optimoptions('fminunc', 'Algorithm','trust-region', 'Display', 'off', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective');

obj = @(p)subprob_LZBZ(p, X, v, gamma, t, b);

tmp = fminunc(obj, reshape(X, S*N, 1), options);

Y = reshape(tmp, S, N);

end
