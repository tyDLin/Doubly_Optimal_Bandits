%%
clear;
clc; 
close all;

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate (c1,...,cN), (g1,...,gN), (q1,....,qS) and (d1,....,dS). 
B = 1;      
C = 1;      
G = 1;      
Q = 1;      

nplayers    = [50; 100];
nresource   = 2;
nbarrier    = 0.5;

ntrials     = 10;    
T           = 5100;        

for di=1:length(nplayers)
    
    errs_BLM  = zeros(T+1, 2); 
    errs_LZBZ = zeros(T+1, 2);
    
    tmp_dist_BLM  = zeros(T+1, ntrials); 
    tmp_dist_LZBZ = zeros(T+1, ntrials); 
    
    params.N = nplayers(di);      % N denotes the number of players
    params.S = nresource;         % S denotes the number of resources
    params.b = B;    
    
    for dn = 1:ntrials
        
        fprintf('%i\t', dn);
    
        %% generate data
        params.c = C*rand(params.N, 1)+1; 
        params.g = G*rand(params.N, 1);
                
        params.q = Q*rand(params.S, 1);  
        params.d = nbarrier*rand(params.S,1);   
        
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
    
        [~, disthist_BLM] = centroid_BLM(X_star, params, optsBLM);

        %% call LZBZ
        optsLZBZ.LZBZ_max_iters  = T;
        optsLZBZ.display         = 0;
        optsLZBZ.displayfreq     = 100;
        optsLZBZ.checkfreq       = 100;
        optsLZBZ.savedisthist    = 1;
        
        [~, disthist_LZBZ] = centroid_LZBZ(X_star, params, optsLZBZ); 
                
        %% set the result at each round
        tmp_dist_BLM(:, dn)  = disthist_BLM;
        tmp_dist_LZBZ(:, dn) = disthist_LZBZ;
        
    end
    
    fprintf('\n');
    
    errs_BLM(:, 1)  = mean(tmp_dist_BLM, 2); 
    errs_BLM(:, 2)  = std(tmp_dist_BLM, 0, 2); 
    errs_LZBZ(:, 1) = mean(tmp_dist_LZBZ, 2); 
    errs_LZBZ(:, 2) = std(tmp_dist_LZBZ, 0, 2); 
    
    round = 1:(T/10):T; 

    figure; 
    errorbar(round, errs_LZBZ(round, 1), 0.5*errs_LZBZ(round, 2), '-d', 'LineWidth', 3, 'MarkerSize', 15);
    hold on
    errorbar(round, errs_BLM(round, 1), 0.5*errs_BLM(round, 2), '-*', 'LineWidth', 3, 'MarkerSize', 15);
    hold off
    legend('Our Algorithm', 'Multi-Agent FKM', 'Location', 'northeast', 'Orientation', 'vertical');
        
    set(gca, 'FontSize', 20);
    xlabel('Iteration Count');
    ylabel('$\|\hat{x} - x^\star\|/(1+\|x^\star\|)$', 'interpreter', 'latex');
    xlim([0 T])
    ylim([0 5])
    yticks([0 1 2 3 4 5])
    title(['N=', num2str(params.N)]);

    path = sprintf('../figs/KA_%d', params.N); 
    saveas(gcf, path, 'epsc');

end
