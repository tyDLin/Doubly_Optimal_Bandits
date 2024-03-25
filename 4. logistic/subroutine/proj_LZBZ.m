function Y = proj_LZBZ(X, v, gamma, mu, t, d)  

options = optimoptions('fminunc', 'Algorithm','trust-region', 'Display', 'off', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective');

x0 = X;
obj = @(p)subprob_LZBZ(p, X, v, gamma, mu, t, d);

Y = fminunc(obj, x0, options);

end
