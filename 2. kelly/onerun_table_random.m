%%************************************************************************* 
%% Call different algorithms to solve Kelly auction problems
%%*************************************************************************

%%
clear;
clc; 
close all; 

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate (c1,...,cN), (g1,...,gN) and (d1,....,dS). 
B = 1;  % capacity
G = 1;  % upper bound for marginal revenue
Q = 1;  % upper bound for avaliable source. 

nplayers  = [10; 20; 50; 100];
nresource = [2, 5];
nbarrier  = [0.5; 1];

ntrials = 10;   % number of trials. 
T = 5100;       % time horizon. 

errs_BLM  = zeros(length(nplayers), length(nresource), length(nbarrier), 2); 
errs_LZBZ = zeros(length(nplayers), length(nresource), length(nbarrier), 2); 

for di=1:length(nplayers)
    for dj=1:length(nresource)
        for dk=1:length(nbarrier)
                
            tmp_err = zeros(ntrials, 2); 

            for dn = 1:ntrials

                fprintf('%i\t', dn);

                %% generate data
                params.N = nplayers(di);          % N denotes the number of players
                params.S = nresource(dj);         % S denotes the number of resources
            
                % parameters for players
                params.b = B;
                params.g = G*rand(params.N, 1);

                % parameters for resources
                params.q = Q*rand(params.S, 1);  
                params.d = nbarrier(dk)*rand(params.S,1);   

                %% call operator extrapolation
                optsOE.nIter            = 1000;
                optsOE.lam              = 0.5;
                optsOE.gam              = 0.01;
                optsOE.display          = 0;
                optsOE.displayfreq      = 50;
                X_star = centroid_OE(params, optsOE);
                
                %% call BLM
                optsBLM.BLM_max_iters   = T;
                optsBLM.display         = 0;
                optsBLM.displayfreq     = 100;
                optsBLM.checkfreq       = 100;

                [X_BLM, ~] = centroid_BLM(X_star, params, optsBLM);
                err_BLM = norm(X_BLM-X_star)/(1 + norm(X_star)); 

                %% call LZBZ
                optsLZBZ.LZBZ_max_iters  = T;
                optsLZBZ.display         = 0;
                optsLZBZ.displayfreq     = 100;
                optsLZBZ.checkfreq       = 100;

                [X_LZBZ, ~] = centroid_LZBZ(X_star, params, optsLZBZ); 
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
    for dj=1:length(nresource)
        for dk=1:length(nbarrier)
            
            N = nplayers(di);        
            S = nresource(dj);
            D = nbarrier(dk); 
            fprintf('(%i, %i, %3.2f) & %0.1e (%0.1e) & %0.1e (%0.1e) \n', ...
                    N, S, D, errs_BLM(di, dj, dk, 1), errs_BLM(di, dj, dk, 2), ...
                    errs_LZBZ(di, dj, dk, 1), errs_LZBZ(di, dj, dk, 2));
        end
    end
end