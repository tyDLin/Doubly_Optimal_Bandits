%%************************************************************************* 
%% Call different algorithms to solve regularized logistic regression problems
%%*************************************************************************

%%
clear;
clc; 
close all; 

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting.  
ndatasets = {'a9a', 'mushrooms', 'news20', 'splice', 'svmguide3', 'w8a'};

T = 20100;   % Maximum Iteration. 

for di=1:length(ndatasets) 
        
    %% generate data. 
    dataset_name = ndatasets{di};
    
    fprintf('\nProcessing (%d/%d) dataset: %s\n', di, length(ndatasets), dataset_name);
    data_path = './data/';
    load([data_path dataset_name '.mat']);
    
    params.A    = full(samples);
    [m, N]      = size(samples); 
    params.b    = labels;
    params.mu   = 0.001; 
    params.d    = 1/N; 
    
    %% call accelerated gradient descent. 
    optsAGD.nIter           = 500;
    optsAGD.display         = 0;
    optsAGD.displayfreq     = 100;
    X_star                  = centroid_AGD(params, optsAGD);

    %% call BLM
    optsBLM.BLM_max_iters   = T;
    optsBLM.display         = 0;
    optsBLM.displayfreq     = 100;
    optsBLM.checkfreq       = 100;
    optsBLM.savedisthist    = 1;
    optsBLM.savetimehist    = 1;
    [~, disthist_BLM] = centroid_BLM(X_star, params, optsBLM);

    %% call LZBZ
    optsLZBZ.LZBZ_max_iters = T;
    optsLZBZ.display        = 0;
    optsLZBZ.displayfreq    = 100;
    optsLZBZ.checkfreq      = 100;
    optsLZBZ.savedisthist   = 1;
    optsLZBZ.savetimehist   = 1;
    [~, disthist_LZBZ] = centroid_LZBZ(X_star, params, optsLZBZ);  

    %% plot the figures
    round = 1:(T/10):T; 

    figure; 
    plot(round, disthist_LZBZ(round), '-d', 'LineWidth', 3, 'MarkerSize', 15);
    hold on
    plot(round, disthist_BLM(round), '-*', 'LineWidth', 3, 'MarkerSize', 15);
    hold off
    legend('Our Algorithm', 'Multi-Agent FKM', 'Location', 'northeast', 'Orientation', 'vertical');
        
    set(gca, 'YScale','log');
    set(gca, 'FontSize', 20);
    xlabel('Iteration Count');
    ylabel('$\|\hat{x} - x^\star\|/(1+\|x^\star\|)$', 'interpreter', 'latex');
    xlim([0 T])
    ylim([0 1])
    yticks([0 0.2 0.4 0.6 0.8 1])
    title([dataset_name]);

    path = sprintf('../figs/LR_%s', dataset_name); 
    saveas(gcf, path, 'epsc');
end