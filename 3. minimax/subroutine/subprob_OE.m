function f = subprob_OE(X, X1, F0, F1, gam, lam)

[S, N] = size(X1);
% reshape matrix to vector

X = reshape(X, S*N, 1);
X1 = reshape(X1, S*N, 1);

F0 = reshape(F0, S*N, 1);
F1 = reshape(F1, S*N, 1);

f = gam * (F1+lam*(F1-F0))'*X + 1/2*sum((X1-X).^2);


end