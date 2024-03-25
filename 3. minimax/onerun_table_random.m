%%
clear;
clc; 
close all; 

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate A
ndim = [10; 20; 50; 100];
nrange = [0.5; 1]; 

ntrials = 10;           
T       = 5100; 

errs_BLM  = zeros(length(ndim), length(nrange), 2); 
errs_LZBZ = zeros(length(ndim), length(nrange), 2); 

for di=1:length(ndim)
    for dj=1:length(nrange)
        
        tmp_err = zeros(ntrials, 2);

        params.N = 2;           % N denotes the number of players
        params.S = ndim(di);    % S denotes the number of resources
        params.b = nrange(dj); 
        params.mu = 0.01; 

        for dn = 1:ntrials

            fprintf('%i\t', dn);

            %% generate data
            params.A = sprand(params.S, params.S, 0.5); 
    
            %% call OGDA
            optsOE.nIter            = 1000;
            optsOE.lam              = 0.5;
            optsOE.gam              = 0.001;
            optsOE.display          = 0;
            optsOE.displayfreq      = 100;
            X_star                  = centroid_OE(params, optsOE);
                
            %% call BLM
            optsBLM.BLM_max_iters   = T;
            optsBLM.display         = 0;
            optsBLM.displayfreq     = 100;
            optsBLM.checkfreq       = 100;
            optsBLM.savedisthist    = 1;
            
            [X_BLM, dist_BLM] = centroid_BLM(X_star, params, optsBLM);
            err_BLM = norm(X_BLM-X_star)/(1 + norm(X_star)); 

            %% call LZBZ
            optsLZBZ.LZBZ_max_iters  = T;
            optsLZBZ.display         = 0;
            optsLZBZ.displayfreq     = 100;
            optsLZBZ.checkfreq       = 100;
            optsLZBZ.savedisthist    = 1;

            [X_LZBZ, dist_LZBZ] = centroid_LZBZ(X_star, params, optsLZBZ);
            err_LZBZ = norm(X_LZBZ-X_star)/(1 + norm(X_star));  

            %% set the result at each round
            tmp_err(dn, 1) = err_BLM;
            tmp_err(dn, 2) = err_LZBZ;
        end

        fprintf('\n');

        errs_BLM(di, dj, 1)   = mean(tmp_err(:, 1));
        errs_BLM(di, dj, 2)   = std(tmp_err(:, 1)); 
        errs_LZBZ(di, dj, 1)  = mean(tmp_err(:, 2)); 
        errs_LZBZ(di, dj, 2)  = std(tmp_err(:, 2));             
    end
end

%% print results
fprintf('\n'); 

fprintf('print results\n'); 

for di=1:length(ndim)
    for dj=1:length(nrange)
            n = ndim(di);        
            b = nrange(dj); 
            fprintf('(%i, %3.2f) & %0.1e %0.1e & %0.1e %0.1e \n', ...
                    n, b, errs_BLM(di, dj, 1), errs_BLM(di, dj, 2), ...
                    errs_LZBZ(di, dj, 1), errs_LZBZ(di, dj, 2));
    end
end