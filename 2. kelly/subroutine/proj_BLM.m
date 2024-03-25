function Y = proj_BLM(X, b)

[S, N] = size(X);
X = reshape(X, S*N, 1);
A = zeros(N, S*N);
for i = 1:N
    A(i, (1+(i-1)*S):(i*S)) = ones(1, S);
end

options = optimoptions(@fmincon, 'Display', 'off', 'Algorithm', 'sqp');

Y = fmincon(@(p) sum((p-X).^2), X, A, b*ones(N,1), [], [], zeros(N*S,1), [], [], options);
Y = reshape(Y, S, N);

end

