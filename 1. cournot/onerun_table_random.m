%%************************************************************************* 
%% Call different algorithms to solve Cournot competition problems
%%*************************************************************************

%%
clear;
clc; 
close all; 

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate (c1,c2,...,cN)
B = 1;
C = 1; 
nplayers = [10; 20; 50; 100];
nmarket = [10; 20];
inv_demand = [0.05; 0.1];  

ntrials = 10;  % number of trials. 
T = 20000;      % number of iterations. 

errs_BLM  = zeros(length(nplayers), length(nmarket), length(inv_demand), 2); 
errs_LZBZ = zeros(length(nplayers), length(nmarket), length(inv_demand), 2);

for di=1:length(nplayers)
    for dj=1:length(nmarket)
        for dk=1:length(inv_demand)
                
            tmp_err  = zeros(ntrials, 2); 
                
            for dn = 1:ntrials
                    
                fprintf('%i\t', dn);
                    
                %% generate data
                N = nplayers(di);        % N denotes the number of players
                a = nmarket(dj);         % a denotes the market size
                b = inv_demand(dk);      % b denotes the inverse demand
                c = C*rand(N, 1); 
                    
                %% call quadprog
                options = optimoptions('quadprog','Display','off');
                H = (b/2) * (ones(N,N) + eye(N, N));
                f = c - a; 
                X_star = quadprog(H, f, [], [], [], [], zeros(N,1),B*ones(N,1), zeros(N,1), options);
                    
                %% call BLM
                optsBLM.BLM_max_iters   = T;
                optsBLM.display         = 0;
                optsBLM.displayfreq     = 10;
                optsBLM.checkfreq       = 10; 
    
                [X_BLM, ~] = centroid_BLM(X_star, a, b, c, B, optsBLM);
                err_BLM = norm(X_BLM-X_star)/(1 + norm(X_star));
                    
                %% call LZBZ
                optsLZBZ.LZBZ_max_iters  = T;
                optsLZBZ.display         = 0;
                optsLZBZ.displayfreq     = 10;
                optsLZBZ.checkfreq       = 10;
    
                [X_LZBZ, ~] = centroid_LZBZ(X_star, a, b, c, B, optsLZBZ); 
                err_LZBZ = norm(X_LZBZ-X_star)/(1 + norm(X_star));  
                    
                %% set the result at each round
                tmp_err(dn, 1) = err_BLM;
                tmp_err(dn, 2) = err_LZBZ;
            end
                
            fprintf('\n');
                
            errs_BLM(di, dj, dk, 1)   = mean(tmp_err(:, 1));
            errs_BLM(di, dj, dk, 2)   = std(tmp_err(:, 1)); 
            errs_LZBZ(di, dj, dk, 1)  = mean(tmp_err(:, 2)); 
            errs_LZBZ(di, dj, dk, 2)  = std(tmp_err(:, 2)); 
        end
    end
end

%% print results
fprintf('\n'); 

fprintf('print results\n'); 
for di=1:length(nplayers)
    for dj=1:length(nmarket)
        for dk=1:length(inv_demand)
            N = nplayers(di);        
            a = nmarket(dj);         
            b = inv_demand(dk);            
            fprintf('(%i, %i, %3.2f) & %0.1e (%0.1e) & %0.1e (%0.1e) \n', ...
                N, a, b, errs_BLM(di, dj, dk, 1), errs_BLM(di, dj, dk, 2), ...
                errs_LZBZ(di, dj, dk, 1), errs_LZBZ(di, dj, dk, 2));  
        end
    end
end