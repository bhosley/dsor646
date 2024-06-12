from MDP import MDP_Tiled

import numpy as np
from time import perf_counter
from tiles3 import IHT
import collections

class SemiGradSARSA(MDP_Tiled):
    name = "Semi-gradient n-step SARSA"

    def __init__(self, env, n=1, **kwargs) -> None:
        super().__init__(env, max_size = 2**8, **kwargs)
        self.algorithm_name = f"Semi-gradient {n}-step SARSA"
        self.nm1 = n-1

    def train(self, num_replications, num_episodes, verbose=False):
        time_start = perf_counter()
        for z in range(num_replications):
            # theta vector of basis function weights
            w = self.qinit*np.ones((self.max_size,1))/self.num_tiles
            C = np.zeros((self.max_size,1))     # eligibility trace vector
            iht_VFA = IHT(self.max_size)        # integer hash table

            print(f"\nSemi-grad {self.nm1+1}-step SARSA("
                 +f"alpha_a={self.alpha_a:<3.2f},alpha_b={self.alpha_b:<3.2f},"
                 +f"eps_a={self.eps_a:<3.2f},eps_b={self.eps_b:<3.2f}, rep {z}...")

            for m in range(num_episodes): # M
                terminated,truncated = False,False                  # Episode complete flags
                Gm = 0                                              # Episode reward
                np.random.seed(int(m + 1e6*z + 1e5*self.offset))    # Set rng seed
                state = self.env.reset(seed=int(z + 1e6 + m + 1e5*self.offset))[0]
                state_queue = collections.deque([state])            # State record array

                # select action based on epsilon-greedy exploration mechanism
                action = self.get_action(state,(w,iht_VFA),self.epsilon(m))

                action_queue = collections.deque([action])      # Action array, record of actions
                reward_queue = collections.deque([])            # Reward array, record of rewards

                # SARSA main loop (first nm1 transitions)
                for _ in range(self.nm1):
                    # Apply action and observe system information 
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    state_queue.append(next_state)      # Append next state to state array
                    reward_queue.append(reward)         # Append reward to reward array
                    Gm += reward                        # Update episode cumulative reward

                    # select action based on epsilon-greedy exploration mechanism
                    next_action = self.get_action(state,(w,iht_VFA),self.epsilon(m))
                    action_queue.append(next_action)

                # SARSA main loop (> first mm1 transitions until end of episode 
                while not(terminated or truncated):
                    # apply action and observe system information 
                    next_state, reward, terminated, truncated, _ = self.env.step(action_queue[-1])
                    state_queue.append(next_state)      # Append next state to state array
                    reward_queue.append(reward)         # Append reward to reward array
                    Gm += reward                        # Update episode cumulative reward

                    # select action based on epsilon-greedy exploration mechanism
                    next_action = self.get_action(state,(w,iht_VFA),self.epsilon(m))
                    action_queue.append(next_action)

                    # Temporal-Difference learning mechanism
                    qhat = (np.dot(reward_queue,self.gamma**np.array(range(len(reward_queue)))) 
                            + (1-terminated)*self.gamma**len(reward_queue)
                            *self.Qbar(next_state,next_action,w,iht_VFA))
                    Δ = qhat - self.Qbar(state_queue[0],action_queue[0],w,iht_VFA)  # TD error

                    # Update w vector
                    active_tiles = self.gradQbar(state_queue[0],action_queue[0],iht_VFA)
                    C[active_tiles] += 1 # update state-action counter 
                    w[active_tiles] += self.alpha(C[active_tiles])*Δ

                    state_queue.popleft()   #
                    action_queue.popleft()  # update by removing oldest values
                    reward_queue.popleft()  #

                # SARSA main loop (episode complete, updates from last nm1 transitions)
                while len(reward_queue)>0:
                    # Temporal Difference learning mechanism
                    qhat = np.dot(reward_queue, self.gamma**np.array(range(len(reward_queue))) )
                    Δ = qhat - self.Qbar(state_queue[0],action_queue[0],w,iht_VFA)  # TD error

                    # Update w vector
                    active_tiles = self.gradQbar(state_queue[0], action_queue[0],iht_VFA)
                    C[active_tiles] += 1 # update state-action counter 
                    w[active_tiles] += self.alpha(C[active_tiles])*Δ

                    # update state, action, and reward arrays by removing oldest values
                    state_queue.popleft()   #
                    action_queue.popleft()  # update by removing oldest values
                    reward_queue.popleft()  #

                if verbose: print(f"In Episode: {m}, Cumulative reward: {Gm}")
                self.Gzm.append((z,Gm))

                # test current policy (as represented by current Q) every test_freq episodes
                if m % self.test_freq == 0:
                    mean, hw = self.evaluate_policy((w, iht_VFA), self.num_test_reps)
                    self.GzmTest.append((z, m, mean, hw))

                    # update best scores if necessary
                    if verbose:
                        self.update_and_print(m, mean, hw, w, iht_VFA)
                    else:
                        self.update_best_scores(mean, hw, w, iht_VFA)

                self.end_episode_callbacks()

            # last test of current algorithm replication
            mean, hw = self.evaluate_policy((w, iht_VFA), self.num_test_reps)
            self.GzmTest.append((z, num_episodes, mean, hw))

            # update best EETDR scores if necessary
            if verbose:
                self.update_and_print(num_episodes, mean, hw, w, iht_VFA)
            else:
                self.update_best_scores(mean, hw, w, iht_VFA)

        time_elapsed = perf_counter() - time_start
        print(f"Executed {num_replications} algorithm reps in {time_elapsed:0.4f} seconds.")

        self.total_training_reps += num_replications
        self.total_episodes += num_episodes
        # Update execution time record
        total_time = (self.total_training_reps * self.avg_execution_time) + time_elapsed
        self.avg_execution_time = total_time / self.total_training_reps


class SemiGradSARSA_boltzmann(SemiGradSARSA):
    name = "Semi-gradient n-step SARSA"

    def __init__(self, env, n=1, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.algorithm_name = f"Semi-gradient {n}-step SARSA with Boltzmann Exploration"
        self.nm1 = n-1
        self.tau            = kwargs.get('tau',1.0)
        self._tau_          = self.tau
        self.cooling_rate   = kwargs.get('cooling_rate',0.000001)


    #@override(SemiGradSARSA) # Needs Python >3.12
    def get_action(self, state, policy, epsilon=0) -> int:
        w, iht = policy
        Qvals = np.array([self.Qbar(state,a,w,iht) for a in range(self.num_actions)])
        exp_q = np.clip(np.exp(Qvals / self.tau), 1e-10, 1e10)
        self.tau = self.tau * (1-self.cooling_rate) # Apply Cooling
        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(range(self.num_actions), p=probabilities)
        return action

    #@override(MDPBase) # Needs Python >3.12
    def end_episode_callbacks(self) -> None:
        self.tau = self._tau_