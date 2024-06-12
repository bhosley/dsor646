from MDP import MDPBase

import numpy as np
from time import perf_counter

class LSPI(MDPBase):
    """Least-Squares Policy Iteration (LSPI) algorithm for reinforcement learning.

    Additional Attributes:
        eta (float): The regularization parameter for the least-squares update, default is 1e-6.
        mem_buffer_size (int): The size of the memory experience replay buffer, default is 2048.
        n_batch_size (int): The number of experiences needed to perform a batch update of the 
                value-function approximation, default is 512.
        num_state_dims (int): The number of dimensions in the state space of the environment.

    Args:
        env: The environment in which the agent will be trained.
        **kwargs: Additional keyword arguments for algorithm-specific parameters.
    """
    name = "Least Squares Policy Iteration"

    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.algorithm_name     = self.name
        self.eta                = kwargs.get('eta',1e-6)                # regularization parameter
        self.mem_buffer_size    = kwargs.get('mem_buffer_size', 2**11)  # experience replay buffer
        self.n_batch_size       = kwargs.get('n_batch_size', 2**9)      # num exps batch update
        self.num_state_dims     = len(env.observation_space.low)        # state space dims
        self.best_scores = [{'ETDR': -np.inf, 'ETDR_hw': np.inf, 'w': None}
                            for _ in range(self.num_best_scores)]

        """ Tunable Hyperparameters with new defaults"""
        self.alpha_a    = kwargs.get('alpha_a',0.25)    # Learning Rate
        self.alpha_b    = kwargs.get('alpha_b',0.75)    # Learning Rate
        self.eps_a      = kwargs.get('eps_a',1.0)       # Eps-Greedy stepsize rule
        self.eps_b      = kwargs.get('eps_b',0.25)      # Eps-Greedy stepsize rule
        self.qinit      = kwargs.get('qinit',1.0)
        self.gamma      = kwargs.get('gamma',0.99)      # discount rate

    #@override(MDPBase) # Needs Python >3.12
    def phi(self,s,a):
        sa = np.hstack(((s-self.Slow)/self.Srange,((s-self.Slow)/self.Srange)**2,a-1))
        sa2 = np.matrix(sa).T*np.matrix(sa)
        up_tri_sa2 = np.triu(sa2)
        sa2_flat = up_tri_sa2[np.triu_indices_from(up_tri_sa2)]
        return np.hstack((1, sa, sa2_flat))

    def batchPhi(self,s_batch,a_batch):
        return np.array([self.phi(s, a) for s, a in zip(s_batch, a_batch)])

    def Qbar(self,s,a,w):
        return np.dot(w,self.phi(s,a))

    def batchArgmaxQbar(self,s_arrays, w):
        # Compute Qvals for each combination of s and a
        Qvals = np.array([[self.Qbar(s, a, w) for a in range(self.num_actions)] for s in s_arrays])
        # Find the maximum value for each row (i.e., each s array)
        max_actions = np.argmax(Qvals, axis=1)
        return max_actions

    def argmaxQbar(self,s,w):
        Qvals = [self.Qbar(s,a,w) for a in range(self.num_actions)]
        return np.argmax(Qvals)

    #@override(MDPBase)
    def update_best_scores(self, mean, hw, w):
    # Find the first score that mean is greater than 
        for i in range(len(self.best_scores)):
            if mean-hw > self.best_scores[i]['ETDR'] - self.best_scores[i]['ETDR_hw']:
                # Shift scores and parameters
                self.best_scores.insert(i, {'ETDR': np.copy(mean), 'ETDR_hw': np.copy(hw), 
                                       'w': np.copy(w)})
                self.best_scores.pop()   # Remove least best score
                return True
        return False

    #@override(MDPBase) # Needs Python >3.12
    def evaluate_policy(self, w, num_reps, seed_mult=1):
        # initialize_test_data_structure
        test_data = np.zeros((num_reps))
        # run num_test_reps replication per test
        for rep in range(num_reps):
            # initialize episode conditions
            terminated = False
            truncated = False
            # Initialize episode reward
            Gtest = 0
            # Initialize the system by resetting the environment, obtain state var
            state = self.env.reset(seed=seed_mult*1000+rep)[0]
            while not (terminated or truncated):
                # select action with highest q-value
                action = self.argmaxQbar(state,w)
                # apply action and observe system information
                state, reward, terminated, truncated, _ = self.env.step(action)
                # update episode cumulative reward
                Gtest += reward
            test_data[rep] = Gtest
        mean, hw = self.confinterval(test_data)
        return mean, hw

    #@override(MDPBase) # Needs Python >3.12
    def find_superlative(self, num_test_reps=30):
        # initialize list of means and half-widths for testing top policies
        mean_values, hw_values = [], []
        # loop through top policies stored in best_scores to find superlative policy
        for i, score in enumerate(self.best_scores):
            mean, hw = self.evaluate_policy(score['w'], num_test_reps, 2)
            print(f"\nBest VFA ({self.ordinal(i+1)}) test... EETDR CI: {mean:>6.1f} +/- {hw:4.1f}")
            mean_values.append(mean)
            hw_values.append(hw)
        # determine superlative policy and record its mean and half-width
        indBestCILB = np.argmax(np.array(mean_values)-np.array(hw_values))
        maxETDR = mean_values[indBestCILB]
        maxETDRhw = hw_values[indBestCILB]
        return indBestCILB, maxETDR, maxETDRhw

    def train(self, num_replications, num_episodes, verbose=False):
        time_start = perf_counter()
        num_features = len(self.phi(self.env.reset()[0],0))  # number features

        for z in range(num_replications):
            mem_buffer_counter = 0  # initialize memory replay buffer counter
            # initialize tuple memory buffer components
            mem_buffer_states = np.zeros((self.mem_buffer_size, self.num_state_dims),
                                         dtype=np.float32)
            mem_buffer_next_states = np.zeros((self.mem_buffer_size, self.num_state_dims),
                                              dtype=np.float32)
            mem_buffer_actions = np.zeros(self.mem_buffer_size, dtype=np.int32)
            mem_buffer_rewards = np.zeros(self.mem_buffer_size, dtype=np.float32)
            mem_buffer_done_flags = np.zeros(self.mem_buffer_size, dtype=np.int32)

            # initialize theta vector of basis function weights
            w = np.zeros(num_features)

            print(f"\nLSPI(alpha_a={self.alpha_a:<3.2f}, alpha_b={self.alpha_b:<3.2f},"
                + f"eps_a={self.eps_a:<3.2f}, eps_b={self.eps_b:<3.2f} rep {z}...")

            for m in range(num_episodes): # M
                np.random.seed(int(m + 1e6*z + 1e5*self.offset))    # Set rng seed
                state = self.env.reset(seed=int(z + 1e6 + m + 1e5*self.offset))[0]
                terminated,truncated = False,False                  # Episode complete flags
                G = 0                                               # Episode reward

                # LSPI main loop, implementing trajectory following state sampling
                while not(terminated or truncated):
                    # select action based on epsilon-greedy exploration mechanism
                    if np.random.rand() > self.epsilon(m):      # With probability epsilon:
                        action = self.argmaxQbar(state,w)       # Act greedy, using Qbar
                    else:
                        action = self.env.action_space.sample() # Act randomly to explore

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    G += reward                     # Update episode cumulative reward
                    state = np.copy(next_state)     # Update state

                    # store unite experience in memory replay buffer, increment counter 
                    index = mem_buffer_counter % self.mem_buffer_size 
                    mem_buffer_states[index, :] = state 
                    mem_buffer_actions[index] = action 
                    mem_buffer_rewards[index] = reward 
                    mem_buffer_next_states[index, :] = next_state 
                    mem_buffer_done_flags[index] = terminated 
                    mem_buffer_counter += 1

                # update policy -- select n_batch_size experiences and w solving regression problem
                if mem_buffer_counter >= self.n_batch_size:
                    # sample memory buffer, collecting a batch of SARnSd transitions
                    # determine indexes of experiences used to update policy 
                    max_memory = min(self.mem_buffer_size, mem_buffer_counter)
                    batch_indexes = np.random.choice(np.arange(max_memory), self.n_batch_size, 
                                                     replace=False)

                    # specify batches
                    states_batch = mem_buffer_states[batch_indexes, :]
                    actions_batch = mem_buffer_actions[batch_indexes] 
                    rewards_batch = mem_buffer_rewards[batch_indexes] 
                    next_states_batch = mem_buffer_next_states[batch_indexes, :]
                    done_flags_batch = mem_buffer_done_flags[batch_indexes]

                    # compute matrices and vectors for normal equation
                    Phi_curr = np.matrix(self.batchPhi(states_batch, actions_batch))
                    next_actions_batch = self.batchArgmaxQbar(next_states_batch, w)
                    Phi_next = np.matrix(self.batchPhi(next_states_batch, next_actions_batch))
                    r = np.matrix(rewards_batch).T
                    # ensure terminal state-action pairs have zero value
                    Phi_next[done_flags_batch==1,1:] = 0
                    # determine w by solving normal equation with L2 regularization 
                    what = np.ravel(np.linalg.solve(Phi_curr.T*(Phi_curr-self.gamma*Phi_next)
                                                    +self.eta*np.eye(Phi_curr.shape[1]), Phi_curr.T*r))
                    #                                +self.eta, Phi_curr.T*r))
                    #+self.eta*np.eye(Phi_curr.shape[0]), Phi_curr.T*r))
                    #what = np.ravel(np.linalg.lstsq(Phi_curr.T*(Phi_curr-self.gamma*Phi_next)
                    #                                +self.eta, Phi_curr.T*r)[0])
                    # Polyak averaging
                    w = (1-self.alpha(m))*w + self.alpha(m)*what

                if verbose: print(f"In Episode: {m}, Cumulative reward: {G}")
                self.Gzm.append((z,G))

                # test current policy (as represented by current Q) every test_freq episodes
                if m % self.test_freq == 0:
                    mean, hw = self.evaluate_policy(w, self.num_test_reps)
                    self.GzmTest.append((z, m, mean, hw))

                    # update best scores if necessary
                    if verbose:
                        self.update_and_print(m, mean, hw, w)
                    else:
                        self.update_best_scores(mean, hw, w)

            # last test of current algorithm replication
            mean, hw = self.evaluate_policy(w, self.num_test_reps)
            self.GzmTest.append((z, num_episodes, mean, hw))

            # update best EETDR scores if necessary
            if verbose:
                self.update_and_print(num_episodes, mean, hw, w)
            else:
                self.update_best_scores(mean, hw, w)

        time_elapsed = perf_counter() - time_start
        print(f"Executed {num_replications} algorithm reps in {time_elapsed:0.4f} seconds.")

        self.total_training_reps += num_replications
        self.total_episodes += num_episodes
        # Update execution time record
        total_time = (self.total_training_reps * self.avg_execution_time) + time_elapsed
        self.avg_execution_time = total_time / self.total_training_reps



#import gymnasium as gym
#env = gym.make('LunarLander-v2')
#lspi = LSPI(env)
#lspi.train(num_replications=5, num_episodes=100)
#lspi.get_results()
#lspi.show_results()
#lspi.display_best_policy()
#   KeyError: 'Q'
#print(lspi.find_superlative())