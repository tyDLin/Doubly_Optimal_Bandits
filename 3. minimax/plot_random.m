%%
clear;
clc; 
close all; 

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate A
ndim = [50; 100];

ntrials = 10;           
T       = 5100; 

for di=1:length(ndim)
    
    err_BLM  = zeros(T+1, 2); 
    err_LZBZ = zeros(T+1, 2);
    
    tmp_dist_BLM  = zeros(T+1, ntrials); 
    tmp_dist_LZBZ = zeros(T+1, ntrials); 

    params.N = 2;           % N denotes the number of players
    params.S = ndim(di);    % S denotes the number of resources
    params.b = 1; 
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

        [~, disthist_BLM] = centroid_BLM(X_star, params, optsBLM);

        %% call LZBZ
        optsLZBZ.LZBZ_max_iters = T;
        optsLZBZ.display        = 0;
        optsLZBZ.displayfreq    = 100;
        optsLZBZ.checkfreq      = 100;
        optsLZBZ.savedisthist   = 1;

        [~, disthist_LZBZ] = centroid_LZBZ(X_star, params, optsLZBZ);

        %% set the result at each round
        tmp_dist_BLM(:, dn)  = disthist_BLM;
        tmp_dist_LZBZ(:, dn) = disthist_LZBZ;
        
    end

    fprintf('\n');
    
    err_BLM(:, 1)   = mean(tmp_dist_BLM, 2); 
    err_BLM(:, 2)   = std(tmp_dist_BLM, 0, 2); 
    err_LZBZ(:, 1)  = mean(tmp_dist_LZBZ, 2); 
    err_LZBZ(:, 2)  = std(tmp_dist_LZBZ, 0, 2);
    
    round = 1:(T/10):T;

    %% plot the figures
    figure; 
    errorbar(round, err_LZBZ(round, 1), 0.5*err_LZBZ(round, 2), '-d', 'LineWidth', 3, 'MarkerSize', 15);
    hold on
    errorbar(round, err_BLM(round, 1), 0.5*err_BLM(round, 2), '-*', 'LineWidth', 3, 'MarkerSize', 15);
    hold off
    legend('Our Algorithm', 'Multi-Agent FKM', 'Location', 'northeast', 'Orientation', 'vertical');
    
    set(gca, 'YScale','log');
    set(gca, 'FontSize', 20);
    xlabel('Iteration Count');
    ylabel('$\|\hat{x} - x^\star\|/(1+\|x^\star\|)$', 'interpreter', 'latex');
    xlim([0 T])
    ylim([0 1])
    yticks([0 0.2 0.4 0.6 0.8 1])
    title(['n=', num2str(params.S)]);

    path = sprintf('../figs/MM_%d', params.S); 
    saveas(gcf, path, 'epsc');
end