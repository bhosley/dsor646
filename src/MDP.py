import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn import metrics
from abc import abstractmethod


class MDPBase():
    """
    Base class for Markov Decision Process (MDP) algorithms.

    This class provides common functionalities and parameters for reinforcement learning
    algorithms that interact with an environment defined by an MDP. It includes methods
    for handling hyperparameters, discretizing state spaces, and evaluating policies.

    Attributes:
        env: The environment in which the agent will be trained.
        algorithm_name (str): The name of the algorithm.
        eps_a (float): Epsilon-Greedy stepsize rule parameter a, default is 1.0.
        eps_b (float): Epsilon-Greedy stepsize rule parameter b, default is 0.5.
        alpha_a (float): Learning rate parameter a, default is 1.0.
        alpha_b (float): Learning rate parameter b, default is 0.5.
        qinit (float): Initial Q-value, default is 1.0.
        gamma (float): Discount rate, default is 0.999.
        num_actions (int): Number of discrete actions available in the environment.
        Sintervals (int): Number of intervals for discretizing the state space.
        Slow (ndarray): Lower bound of the state space.
        Shigh (ndarray): Upper bound of the state space.
        Srange (ndarray): Range of the state space.
        Sunit (ndarray): Unit size for discretizing the state space.
        SAsize (ndarray): Size of the state-action space.
        test_freq (int): Policy evaluation test frequency, default is 25.
        num_test_reps (int): Number of replications per test, default is 30.
        offset (int): Random number generator seed manual offset, default is 0.
        Gzm (list): List to record each episode's cumulative reward for measuring online performance.
        GzmTest (list): List to record test episode results.
        num_best_scores (int): Number of top scores to keep track of, default is 10.
        best_scores (list): List of dictionaries containing the top scores and corresponding Q-values.
        total_training_reps (int): Total number of training replications.
        avg_execution_time (float): Average execution time per replication.
        total_episodes (int): Total number of episodes.
        qrange (list): Range of Q-values for the environment.

    Methods:
        ordinal(n): Converts a number to its ordinal representation.
        epsilon(n): Computes the epsilon value for the nth episode.
        alpha(n): Computes the alpha value for the nth episode.
        phi(Scont): Converts a continuous state to a discrete state variable representation.
        confinterval(data, alpha=0.05): Computes the confidence interval for the given data.
        update_best_scores(mean, hw, Q): Updates the list of best scores with a new score if it is among the top.
        update_and_print(episode_number, mean, hw, Q): Updates the best scores and prints the current status.
        evaluate_policy(Q, num_reps=30, seed_mult=1): Evaluates the policy using the Q-values over multiple replications.
        title(reps): Generates a title string for the performance results.
        show_results(): Displays the performance results as a plot.
        display_best_policy(): Displays the best policy using a greedy approach as an animation.
        find_superlative(num_test_reps=30): Finds and returns the superlative policy from the list of best scores.
        get_results(): Computes and returns the key performance metrics.

    Abstract Methods:
        name: Returns the name of the algorithm.
        train(num_replications, num_episodes, verbose=False): Trains the agent using the specified number of replications and episodes.
    """

    def __init__(self, env, **kwargs) -> None:
        self.env = env
        self.algorithm_name = kwargs.get('self.algorithm_name', '')

        """ Tunable Hyperparameters """
        self.eps_a      = kwargs.get('eps_a',1.0)   # Eps-Greedy stepsize rule
        self.eps_b      = kwargs.get('eps_b',0.5)   # Eps-Greedy stepsize rule
        self.alpha_a    = kwargs.get('alpha_a',1.0) # Learning Rate
        self.alpha_b    = kwargs.get('alpha_b',0.5) # Learning Rate
        self.qinit      = kwargs.get('qinit',1.0)
        self.gamma      = kwargs.get('gamma',0.999) # discount rate

        """ Discretizing the Space """
        self.num_actions= env.action_space.n        # Discrete Action Space
        discrete_states = 12
        self.Sintervals = discrete_states - 1
        self.Slow       = np.array(env.observation_space.low)
        self.Shigh      = np.array(env.observation_space.high)
        self.Srange     = self.Shigh-self.Slow
        self.Sunit      = self.Srange/self.Sintervals
        self.Ssize      = discrete_states * np.ones(len(env.observation_space.low)).astype(int)
        self.SAsize     = np.append(self.Ssize,self.num_actions)  # state-action space

        """ Testing Parameters """
        self.test_freq = 25 # policy evaluation test frequency (1/test_freq)
        self.num_test_reps = 30 # number of replications per test 
        self.offset = 0 # random number generator seed manual offset 

        """ Data Store """
        # record each episode's cumulative reward to measure online performance
        self.Gzm = [] 
        self.GzmTest = []
        # Hold the Top 10 scores and VEA parameters 
        self.num_best_scores = 10 
        self.best_scores = [{'ETDR': -np.inf, 'ETDR_hw': np.inf, 'Q': None}
                            for _ in range(self.num_best_scores)]
        self.total_training_reps = 0
        self.avg_execution_time = 0.0
        self.total_episodes = 0

        # Set range of q-values for environment (for setting qinit)
        match self.env.spec.id:
            case 'CartPole-v1' : self.qrange = [0,500]
            case 'MountainCar-v0' : self.qrange = [-200,-100]
            case 'LunarLander-v2' : self.qrange = [0,200]
            case _: self.qrange = [-200,200] # A WAG if unknown environment


    def ordinal(self,n):   # create string for ordinal number
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n%10, 4)]
        if 11 <= (n % 100) <= 13:suffix = 'th'
        return str(n) + suffix

    def epsilon(self,n) :
        return self.eps_a/(1+n)**self.eps_b

    def alpha(self,n):
        return self.alpha_a/(1+n)**self.alpha_b

    def phi(self, Scont) : 
        """convert continuous state to discrete 2D state var representation"""
        a = np.round((Scont-self.Slow)/self.Sunit).astype(int)
        return np.clip(a,0,self.Sintervals)

    def confinterval(self, data, alpha=0.05):
        n   = np.size(data)             # number of data points
        se  = np.std(data)/np.sqrt(n)   # standard error
        ts  = t.ppf(1-alpha/2, n-1)     # t-score
        mean = np.mean(data)
        halfwidth = ts*se
        return mean, halfwidth

    def update_best_scores(self, mean, hw, Q):
        # Find the first score that mean is greater than
        for i in range (len(self.best_scores)):
            lower_bound = self.best_scores[i]['ETDR'] - self.best_scores[i]['ETDR_hw']
            if mean - hw > lower_bound:
                # Insert best score and attendant parameters
                self.best_scores.insert(i, {'ETDR': np.copy(mean), 'ETDR_hw': np.copy(hw),
                                            'Q': np.copy(Q)})
                self.best_scores.pop()  # Remove least best score
                return True
        return False

    def update_and_print(self, episode_number, mean, hw, Q):
        # update best scores if necessary
        best_scores = self.best_scores
        if self.update_best_scores(mean,hw,Q):
            print(f"   Test... Episode: {episode_number:>4}, "
                  + f"EETDR CI: {mean:>6.2f} +/- {hw:4.2f} New Top {self.num_best_scores} "
                  + f"EETDR 95CILB -- Q recorded")
        else:
            print(f"   Test... Episode: {episode_number:>4}, "
                  + f"EETDR CI: {mean:>6.2f} +/- {hw:4.2f}\n"
                  + f"*------* Current Top 5 Reliable EETDRs: "
                  + "".join([f"{best_scores[i]['ETDR']:>6.2f}, " for i in range(5)])    )

    def evaluate_policy(self, policy, num_reps=30, seed_mult=1):
        test_data = np.zeros((num_reps))
        for rep in range(num_reps):
            terminated, truncated = False, False
            Gtest = 0       # Episode reward
            state = self.env.reset(seed=seed_mult*1000+rep)[0]  # Reset the environment, get state
            while not (terminated or truncated):
                action = self.get_action(state, policy)         # Eps=0, deterministically
                state, reward, terminated, truncated, _ = self.env.step(action)
                Gtest += reward     # update episode cumulative reward
            test_data[rep] = Gtest
        mean, hw = self.confinterval(test_data)
        return mean, hw

    def title(self,reps):
        s = (f"{self.algorithm_name} Algorithm Performance, {reps} reps, "
             + f"{np.round(self.avg_execution_time,1)} sec/rep\n"
             + f"gamma={self.gamma}, eps_a={self.eps_a}, eps_b={self.eps_b}, g0={self.qinit}, "
             + f"|S|/d={self.Sintervals+1}\n")
        return s

    def show_results(self):
        Z = self.total_training_reps
        M = self.total_episodes

        npGzmTest = np.array(self.GzmTest)
        size = np.shape(npGzmTest)

        TestEETDR = np.reshape(npGzmTest[:,2],(Z,int(size[0]/Z)))

        maxTestEETDR = np.max(TestEETDR,axis=1)
        meanMaxTestEETDR = np.mean(maxTestEETDR)
        maxTestSE = np.std(maxTestEETDR)/np.sqrt(Z)
        maxTestHW = t.ppf(1-0.05/2,Z-1)*maxTestSE

        avgTestEETDR = np.mean(TestEETDR, axis=0)
        avgTestSE = np.std(TestEETDR,axis=0)/np.sqrt(Z)
        avgTestHW = t.ppf(1-0.05/2,Z-1)*avgTestSE

        AULC = [metrics.auc(np.arange(0,M+1,self.test_freq), TestEETDR[z,:])/M for z in range(Z)]
        meanAULC = np.round(np.mean(AULC),1)
        hwAULC = t.ppf(1-0.05/2, Z-1)*np.std(AULC)/np.sqrt(Z)

        _, maxETDR, maxETDRhw = self.find_superlative()

        plt.figure(0)
        X = np.arange(0,M+1,self.test_freq)
        plt.plot(X, avgTestEETDR, marker='o', ms=3, mec='k', linewidth=1, label='Mean EETDR')
        plt.fill_between(X, avgTestEETDR+avgTestHW, avgTestEETDR-avgTestHW, linestyle = '--',
                        linewidth=0.5, facecolor='blue', alpha=0.15, label='95% Halfwidth')
        plt.xlabel('Episode')
        plt.ylabel('Mean Estimated Expected\nTotal Discounted Reward (EETDR)')
        sub_title = (f"Mean Max EETDR: {meanMaxTestEETDR:>6.2f} +/- {maxTestHW:4.2f}, "
                     + f"Mean Time-Avg EETDR: {meanAULC:>6.2f} +/- {hwAULC:4.2f})\n"
                     + f"Superlative Policy EETDR: {maxETDR:>6.2f} +/- {maxETDRhw:4.2f}")
        plt.title(self.title(reps=Z)+sub_title, fontsize =9)
        plt.legend(loc="lower right",fontsize=7)
        plt.xlim([-0.05*M, 1.05*M])
        #plt.ylim([-5,505])
        plt.grid(which='both')
        plt.show()

    def display_best_policy(self):
        """display best policy using greedy-only approach as an animation"""
        # initialize environment
        env = gym.make(self.env.spec.id, render_mode='human')
        indBestCILB,_,_ = self.find_superlative()
        Q = np.copy(self.best_scores[indBestCILB]['Q'])
        # number of reps to show
        num_reps_show = 10
        # initialize test data structure
        test_data = np.zeros((num_reps_show)) # original test_data = np.zeros((num_runs_show))
        # perform test replications
        for rep in range(num_reps_show):
            terminated, truncated = False, False
            # initialize episode reward
            Gtest = 0
            # initialize the system by resetting the environment, obtain state var
            state_continuous = env.reset(seed=1000+rep+self.offset)[0]
            state = self.phi(state_continuous)
            while not (terminated or truncated):
                action = np.argmax(Q[tuple(self.phi(state))])  # select action with highest q-value
                # apply action and observe system information
                state_continuous, reward, terminated, truncated, _ = env.step(action)
                state = self.phi(state_continuous)
                # update episode cumulative reward
                Gtest += reward
            test_data[rep] = Gtest
            print(f"Episode {rep} ETDR: {np.round(Gtest,4)}")
        env.close()
        mean, hw = self.confinterval(test_data)
        print (f"\n Rendered episodes... ETDR CI: {np.round(mean,1)} +/- {np.round(hw,1)}")
        return

    def find_superlative(self, num_test_reps=30):
        # initialize list of means and half-widths for testing top policies
        mean_values, hw_values = [], []
        # loop through top policies stored in best_scores to find superlative policy
        for i, score in enumerate(self.best_scores):
            mean, hw = self.evaluate_policy(score['Q'], num_test_reps,2)
            print(f"\nBest VFA ({self.ordinal (i+1)}) test... \EETDR CI: {mean:>6.2f}+/-{hw:4.2f}")
            mean_values.append(mean)
            hw_values.append(hw)
        # determine superlative policy and record its mean and half-width
        indBestCILB = np.argmax(np.array(mean_values)-np.array(hw_values))
        maxETDR = mean_values[indBestCILB]
        maxETDRhw = hw_values[indBestCILB]
        return indBestCILB, maxETDR, maxETDRhw

    def get_results(self):
        Z = self.total_training_reps
        M = self.total_episodes
        npGzmTest = np.array(self.GzmTest)
        size = np.shape(npGzmTest)
        TestEETDR = np.reshape(npGzmTest[:,2],(Z,int(size[0]/Z)))
        maxTestEETDR = np.max(TestEETDR,axis=1)
        meanMaxTestEETDR = np.mean(maxTestEETDR)
        maxTestSE = np.std(maxTestEETDR)/np.sqrt(Z)
        maxTestHW = t.ppf(1-0.05/2,Z-1)*maxTestSE
        AULC = [metrics.auc(np.arange(0,M+1,self.test_freq), TestEETDR[z,:])/M for z in range(Z)]
        meanAULC = np.round(np.mean(AULC),1)
        hwAULC = t.ppf(1-0.05/2, Z-1)*np.std(AULC)/np.sqrt(Z)
        _, maxETDR, maxETDRhw = self.find_superlative()
        return (maxETDR, maxETDRhw, meanMaxTestEETDR, maxTestHW, meanAULC, hwAULC, 
                self.avg_execution_time)

    @property
    @abstractmethod
    def name(self):
        """ The name of the implemented algorithm. """
        pass

    @abstractmethod
    def get_action(self, state, policy, epsilon=0) -> int:
        """
        Selects an action based on the current state and policy.

        Args:
            state: The current state of the environment.
            policy: The policy to be followed for action selection.
            epsilon (float): The probability of selecting a random action 
                (for epsilon-greedy policies), default is 0, which is deterministic.

        Returns:
            int: The selected action.
        """
        pass

    @abstractmethod
    def train(self, num_replications, num_episodes, verbose=False):
        """
        Trains the agent using the specified number of replications and episodes.

        This method should be implemented by inheritors to define the training
        process of the agent. It typically involves running multiple episodes 
        of interaction with the environment, updating the policy, and optionally 
        logging training progress.

        Args:
            num_replications (int): The number of replications to run for training.
            num_episodes (int): The number of episodes to run per replication.
            verbose (bool): If True, print detailed training progress. Default is False.

        Returns:
            None
        """
        pass

    @abstractmethod
    def end_episode_callbacks(self) -> None:
        """
        A holder function that should execute at the end of each episode
        """
        pass


from tiles3 import tiles, IHT
from copy import deepcopy
#from typing import override # Needs Python >3.12

class MDP_Tiled(MDPBase):
    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.max_size       = kwargs.get('max_size',2**10)  # Tile coding for state-action space
        self.num_tiles      = kwargs.get('num_tiles',4)
        self.scale_factor   = kwargs.get('scale_factor',self.num_tiles/self.Srange)  # for tiles
        self.best_scores    = [{'ETDR': -np.inf, 'ETDR_hw': np.inf, 'w': None, 'iht': None} 
                               for _ in range(self.num_best_scores)]


    #@override(MDPBase) # Needs Python >3.12
    def phi(self,s,a,iht):
        return tiles(iht, self.num_tiles,list(s*self.scale_factor),[a])

    def gradQbar(self,s,a,iht):
        return self.phi(s,a,iht)

    def Qbar(self,s,a,w,iht):
        qhat = 0
        tiles = self.phi(s,a,iht)
        for tile in tiles:
            qhat += w[tile]
        return qhat[0]

    def argmaxQbar(self,s,w,iht):
        Qvals = [self.Qbar(s,a,w,iht) for a in range(self.num_actions)]
        return np.argmax(Qvals)

    #@override(MDPBase) # Needs Python >3.12
    def get_action(self, state, policy, epsilon=0) -> int:
        w, iht = policy
        if np.random.rand() > epsilon:
            return self.argmaxQbar(state,w,iht)
        else:
            return self.env.action_space.sample()

    #@override(MDPBase) # Needs Python >3.12
    def update_best_scores(self, mean, hw, w, iht):
        # Find the first score that mean is greater than 
        for i in range(len(self.best_scores)):
            if mean-hw > self.best_scores[i]['ETDR'] - self.best_scores[i]['ETDR_hw']:
                # Shift scores and parameters
                self.best_scores.insert(i, {'ETDR': np.copy(mean), 'ETDR_hw': np.copy(hw), 
                                            'w': deepcopy(w),'iht': deepcopy(iht)})
                self.best_scores.pop()  # We only want the top scores, so remove the last one
                return True
        return False

    #@override(MDPBase) # Needs Python >3.12
    def find_superlative(self, num_test_reps=30):
        mean_values, hw_values = [], []         # lists of means and half-widths for top policies
        for i, score in enumerate(self.best_scores):
            mean, hw = self.evaluate_policy((score['w'], score['iht']), num_test_reps, 2)
            print(f"\nBest VFA ({self.ordinal(i+1)}) test... EETDR CI: {mean:>6.1f} +/- {hw:4.1f}")
            mean_values.append(mean)
            hw_values.append(hw)
        # determine superlative policy and record its mean and half-width
        indBestCILB = np.argmax(np.array(mean_values)-np.array(hw_values))
        maxETDR = mean_values[indBestCILB]
        maxETDRhw = hw_values[indBestCILB]
        return indBestCILB, maxETDR, maxETDRhw

    #@override(MDPBase) # Needs Python >3.12
    def display_best_policy(self, num_reps_show = 10):
        """display best policy using greedy-only approach as an animation"""
        id = self.env.spec.id
        self.env.close()
        self.env = gym.make(id, render_mode='human')
        indBestCILB,_,_ = self.find_superlative()
        test_data = np.zeros((num_reps_show))

        w = np.copy(self.best_scores[indBestCILB]['w'])
        iht = np.copy(self.best_scores[indBestCILB]['iht'])
        # perform test replications
        for rep in range(num_reps_show):
            terminated, truncated = False, False
            # initialize episode reward
            Gtest = 0
            # initialize the system by resetting the environment, obtain state var
            state = self.env.reset(seed=self.offset*1000+rep)[0]
            while not (terminated or truncated):
                action = self.argmaxQbar(state,w,iht)
                # apply action and observe system information
                state, reward, terminated, truncated, _ = self.env.step(action)
                # update episode cumulative reward
                Gtest += reward
            test_data[rep] = Gtest
            print(f"Episode {rep} ETDR: {np.round(Gtest,4)}")
        self.env.close()
        mean, hw = self.confinterval(test_data)
        print (f"\n Rendered episodes... ETDR CI: {np.round(mean,1)} +/- {np.round(hw,1)}")
        return


class FourierBasis:
    def __init__(self, state_dim, order):
        self.state_dim = state_dim
        self.order = order
        self.coefficients = self.generate_coefficients()

    def generate_coefficients(self):
        # Generate all coefficient vectors with values between 0 and order
        coeffs = np.array(np.meshgrid(
            *([range(self.order+1)]*len(self.state_dim))
            )).T.reshape(-1, len(self.state_dim))
        return coeffs

    def transform(self, state):
        # Compute the Fourier basis functions for a given state
        return np.cos(np.pi * np.dot(self.coefficients, state))
