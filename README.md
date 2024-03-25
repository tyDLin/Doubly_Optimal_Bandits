# Doubly Optimal No-Regret Online Learning with Bandit Feedback

These codes provide implementations of solvers for solving strongly monotone games using multi-agent mirror descent self-concordant barrier bandit learning methods. 

# About

We consider online no-regret learning in unknown games with bandit feedback, where each player can only observe its reward at each time – determined by all players’ current joint action – rather than its gradient. We focus on the class of smooth and strongly monotone games and study optimal no-regret learning therein. Leveraging self-concordant barrier functions, we first construct a new bandit learning algorithm and show that it achieves the single-agent optimal regret under smooth and strongly concave reward functions. We then show that if each player applies this no-regret learning algorithm in strongly monotone games, the joint action converges in the last iterate to the unique Nash equilibrium at an optimal rate. Prior to our work, the best-known convergence rate in the same class of games is suboptimal (achieved by a different algorithm), thus leaving open the problem of optimal no-regret learning algorithms. 

Our results thus settle this open problem and contribute to the broad landscape of bandit game-theoretical learning by identifying the first doubly optimal bandit learning algorithm, in that it achieves (up to log factors) both optimal regret in the single-agent learning and optimal last-iterate convergence rate in the multi-agent learning. We also present preliminary numerical results on several application problems (i.e., Cournot competition, Kelly auction, zero-sum two-agent games and distributed logistic regression problem) to demonstrate the efficacy of our algorithm in terms of iteration count.

# Codes

The MATLAB Implementations on both synthetic and real data are provided.  

# References

W. Ba, T. Lin, J. Zhang and Z. Zhou. Doubly Optimal No-Regret Online Learning in Strongly Monotone Games with Bandit Feedback. ArXiv (https://arxiv.org/abs/2112.02856) and SSRN (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3978421). 
