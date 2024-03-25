%%
clear;
clc; 
close all;

addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate (c_1,c_2,...,c_N)
B = 1;
C = 1; 
nplayers    = [10; 20; 50; 100];
nmarket     = 10;
inv_demand  = 0.05;

ntrials     = 10;           
T           = 20100;         

for di=1:length(nplayers)
    
    err_BLM  = zeros(T+1, 2); 
    err_LZBZ = zeros(T+1, 2);
    time_BLM = zeros(T+1, 2); 
    time_LZBZ = zeros(T+1, 2);
    
    tmp_dist_BLM  = zeros(T+1, ntrials); 
    tmp_dist_LZBZ = zeros(T+1, ntrials); 
    tmp_time_BLM  = zeros(T+1, ntrials); 
    tmp_time_LZBZ = zeros(T+1, ntrials); 
    
    N = nplayers(di);       % N denotes the number of players
    a = nmarket;            % a denotes the market size
    b = inv_demand;         % b denotes the inverse demand
    
    for dn = 1:ntrials
        
        fprintf('%i\t', dn);
    
        %% generate data
        c = C*rand(N, 1); 
        
        %% call quadprog
        options = optimoptions('quadprog', 'Display', 'off');
        H = (b/2) * (ones(N,N) + eye(N, N));
        f = c - a; 
        X_star = quadprog(H, f, [], [], [], [], zeros(N,1),B*ones(N,1), zeros(N,1), options);
 
        %% call BLM
        optsBLM.BLM_max_iters   = T;
        optsBLM.display         = 0;
        optsBLM.displayfreq     = 1;
        optsBLM.checkfreq       = 1;
        optsBLM.savedisthist    = 1;
    
        [~, disthist_BLM] = centroid_BLM(X_star, a, b, c, B, optsBLM);

        %% call LZBZ
        optsLZBZ.LZBZ_max_iters  = T;
        optsLZBZ.display         = 0;
        optsLZBZ.displayfreq     = 1;
        optsLZBZ.checkfreq       = 1;
        optsLZBZ.savedisthist    = 1;
        
        [~, disthist_LZBZ] = centroid_LZBZ(X_star, a, b, c, B, optsLZBZ); 
                
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

    figure; 
    errorbar(round, err_LZBZ(round, 1), err_LZBZ(round, 2), '-d', 'LineWidth', 3, 'MarkerSize', 15);
    hold on
    errorbar(round, err_BLM(round, 1), err_BLM(round, 2), '-*', 'LineWidth', 3, 'MarkerSize', 15);
    hold off
    legend('Our Algorithm', 'Multi-Agent FKM', 'Location', 'northeast', 'Orientation', 'vertical');
    
%    set(gca, 'YScale','log');
    set(gca, 'FontSize', 20);
    xlabel('Iteration Count');
    ylabel('$\|\hat{x} - x^\star\|/(1+\|x^\star\|)$', 'interpreter', 'latex');
    xlim([0 T])
    ylim([0 1])
    yticks([0 0.2 0.4 0.6 0.8 1])
    title(['N=', num2str(N)]);

    path = sprintf('../figs/CC_%d', N); 
    saveas(gcf, path, 'epsc');
end
