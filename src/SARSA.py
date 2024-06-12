import numpy as np
from time import perf_counter
from collections import deque

from MDP import MDPBase

class SARSA(MDPBase):
    name = "n-Step SARSA"

    def __init__(self, env, steps=1, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.steps = steps if type(steps) == int else 2**int(6*steps)
        self.algorithm_name = f'{steps}-step SARSA'

    def train(self, num_replications, num_episodes, verbose=False):
        time_start = perf_counter()
        # perform each algorithm replication
        for z in range(num_replications): # Z
            Q = (self.qinit*np.ones(self.SAsize)*np.diff(self.qrange)[0]+self.qrange[0]) # Q-table
            C = np.zeros(self.SAsize)   # state-action counter

            print(f"\n{self.steps} - step SARSA(alpha_a = {self.alpha_a:<3.2f}, alpha_b="
                  + f"{self.alpha_b:<3.2f}, eps_a={self.eps_a:<3.2f}, eps_b={self.eps_b:<3.2f}, "
                  + f"|S|/d {self.Sintervals}, qinit={self.qinit}, rep {z}...")

            for m in range(num_episodes): # M
                np.random.seed(int(m + 1e6*z + 1e5*self.offset))    # Set rng seed
                terminated,truncated = False,False                  # Episode complete flags
                Gm = 0                                              # Episode reward

                state_continuous = self.env.reset(seed=int(z*1e6+m))[0]
                state = self.phi(state_continuous)      # continuous state to discrete
                state_queue = deque([state])            # Initialize state array, records of states
                cs = np.sum(C[tuple(state)])            # Compute times state has been visited

                # Select action based on epsilon-greedy exploration mechanism
                if np.random.rand() > self.epsilon(cs):     # With probability epsilon:
                    action = np.argmax(Q[tuple(state)])     # Act greedy, using Q
                else:
                    action = self.env.action_space.sample() # Act randomly to explore

                action_queue = deque([action])  # Initialize action array, record of actions
                reward_queue = deque([])        # Initialize reward array, record of rewards

                # SARSA main loop (first nml transitions)
                for _ in range(self.steps - 1):
                    # Apply action and observe system information
                    next_state_cont, reward, terminated, truncated, _ = self.env.step(action)
                    next_state = self.phi(next_state_cont)  # Continuous state to discrete
                    reward_queue.append(reward)             # Append reward to reward array
                    Gm += reward                            # Update episode cumulative reward
                    state_queue.append(next_state)          # Append next state to state array
                    cs = np.sum(C[tuple(next_state)])       # State visit count

                    # Select action based on epsilon-greedy exploration mechanism
                    if np.random.rand() > self.epsilon(cs):             # With probability epsilon:
                        next_action = np.argmax(Q[tuple(next_state)])   # Act greedy, using Q
                    else:
                        next_action = self.env.action_space.sample()    # Act randomly to explore
                    action_queue.append(next_action)

                # SARSA main loop (first nml transitions until end of episode)
                while not (terminated or truncated):
                    # Apply action and observe system information
                    next_state_cont, reward, terminated, truncated, _ = self.env.step(action_queue[-1])
                    next_state = self.phi(next_state_cont)  # Continuous state to discrete
                    state_queue.append(next_state)          # Append next state to state array
                    reward_queue.append(reward)             # Append reward to reward array
                    Gm += reward                            # Update episode cumulative reward
                    cs = np.sum(C[tuple(next_state)])       # State visit count

                    # select action based on epsilon-greedy exploration mechanism
                    if np.random.rand() > self.epsilon(cs):             # With probability epsilon:
                        next_action = np.argmax(Q[tuple(next_state)])   # Act greedy, using Q
                    else:
                        next_action = self.env.action_space.sample()    # Act randomly to explore
                    action_queue.append(next_action)

                    # temporal difference learning mechanism
                    qhat = (np.dot(reward_queue, self.gamma**np.arange(len(reward_queue))) 
                            + (1-terminated) * self.gamma**len(reward_queue) 
                            * Q[tuple(next_state) + (next_action,)])

                    # SARSA main loop (episode complete, update for last nml transitions)
                    last_state_action = tuple(state_queue[-1]) + (action_queue[-1],) #in trajectory
                    while reward_queue:
                        # update state-action counter (for earliest state-action in queue) and Q
                        state_action = tuple(state_queue[0]) + (action_queue[0],)
                        C[state_action] += 1
                        csa = C[state_action]   # Count of state-action pair visits
                        Q[state_action] =(1-self.alpha(csa))*Q[state_action] + self.alpha(csa)*qhat

                        state_queue.popleft()   #
                        action_queue.popleft()  # update by removing oldest values
                        reward_queue.popleft()  #

                        # compute qhat
                        qhat = (np.dot(reward_queue, self.gamma**np.arange(len(reward_queue))) 
                               + (1-terminated)*self.gamma**len(reward_queue)*Q[last_state_action])

                    # record performance
                    if verbose: print(f"In Episode: {m}, Cumulative reward: {Gm}")
                    self.Gzm.append((z, Gm))
                    
                # test current policy (as represented by current Q) every test_freq episodes
                if m % self.test_freq == 0:
                    mean, hw = self.evaluate_policy_test(Q, self.num_test_reps)
                    self.GzmTest.append((z,m, mean, hw))
                    # update best scores if necessary
                    if verbose:
                        self.update_and_print(m,mean,hw,Q)
                    else:
                        self.update_best_scores(mean,hw,Q)

            # last test of current algorithm replication
            mean, hw = self.evaluate_policy_test(Q, self.num_test_reps)
            self.GzmTest.append((z, num_episodes, mean, hw))

            # update best EETDR scores if necessary
            if verbose:
                self.update_and_print(m,mean,hw,Q)
            else:
                self.update_best_scores(mean,hw,Q)

        time_elapsed = perf_counter() - time_start
        print(f"Executed {num_replications} algorithm reps in {time_elapsed:0.4f} seconds.")
        
        self.total_training_reps += num_replications
        self.total_episodes += num_episodes
        # Update execution time record
        total_time = (self.total_training_reps * self.avg_execution_time) + time_elapsed
        self.avg_execution_time = total_time / self.total_training_reps


class ExpectedSARSA(MDPBase):
    name = "Expected SARSA"

    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.algorithm_name = self.name

    def train(self, num_replications, num_episodes, verbose=False):
        time_start = perf_counter()
        for z in range(num_replications):
            Q = (self.qinit*np.ones(self.SAsize)*np.diff(self.qrange)[0]+self.qrange[0]) # Q-table
            C = np.zeros(self.SAsize)

            print(f"Expected SARSA (alpha_a = {self.alpha_a:<3.2f}, alpha_b="
                  f"{self.alpha_b:<3.2f}, eps_a={self.eps_a:<3.2f}, eps_b={self.eps_b:<3.2f}, "
                  f"|S|/d {self.Sintervals}, qinit={self.qinit}, rep {z}...")

            for m in range(num_episodes): # M
                np.random.seed(int(m + 1e6*z + 1e5*self.offset))    # Set rng seed
                terminated,truncated = False,False                  # Episode complete flags
                Gm = 0                                              # Episode reward

                state_continuous = self.env.reset(seed=int(z*1e6 + m))[0]
                state = self.phi(state_continuous)      # continuous state to discrete
                state_queue = deque([state])            # Initialize state array, records of states
                cs = np.sum(C[tuple(state)])            # Compute times state has been visited

                # Select action based on epsilon-greedy exploration mechanism
                if np.random.rand() > self.epsilon(cs):     # With probability epsilon:
                    action = np.argmax(Q[tuple(state)])     # Act greedy, using Q
                else:
                    action = self.env.action_space.sample() # Act randomly to explore

                action_queue = deque([action])  # Initialize action array, record of actions
                reward_queue = deque([])        # Initialize reward array, record of rewards

                # Expected-SARSA main loop
                while not (terminated or truncated):
                    next_state_cont, reward, terminated, truncated, _ = self.env.step(action)
                    next_state = self.phi(next_state_cont)  # Continuous state to discrete
                    state_queue.append(next_state)          # Append next state to state array
                    reward_queue.append(reward)             # Append reward to reward array
                    Gm += reward                            # Update episode cumulative reward
                    cs = np.sum(C[tuple(next_state)])       # State visit count

                    # select action based on epsilon-greedy exploration mechanism
                    if np.random.rand() > self.epsilon(cs):             # With probability epsilon:
                        next_action = np.argmax(Q[tuple(next_state)])   # Act greedy, using Q
                    else:
                        next_action = self.env.action_space.sample()    # Act randomly to explore
                    action_queue.append(next_action)

                    # Compute expected Q-value for the next state
                    action_probabilities = (np.ones(self.env.action_space.n) 
                                            * self.epsilon(cs) / self.env.action_space.n)
                    best_action = np.argmax(Q[tuple(next_state)])
                    action_probabilities[best_action] += 1 - self.epsilon(cs)
                    expected_q = np.sum(action_probabilities * Q[tuple(next_state)])

                    td_target = reward + self.gamma*expected_q
                    state_action = tuple(state_queue[-2]) + (action_queue[-2],)
                    C[state_action] += 1
                    csa = C[state_action]   # Count of state-action pair visits
                    Q[state_action] = (1-self.alpha(csa))*Q[state_action]+self.alpha(csa)*td_target

                    action = next_action

                if verbose: print(f"In Episode: {m}, Cumulative reward: {Gm}")
                self.Gzm.append((z, Gm))

                # test current policy (as represented by current Q) every test_freq episodes
                if m % self.test_freq == 0:
                    mean, hw = self.evaluate_policy_test(Q, self.num_test_reps)
                    self.GzmTest.append((z, m, mean, hw))
                    # update best scores if necessary
                    if verbose:
                        self.update_and_print(m, mean, hw, Q)
                    else:
                        self.update_best_scores(mean, hw, Q)

            # last test of current algorithm replication
            mean, hw = self.evaluate_policy_test(Q, self.num_test_reps)
            self.GzmTest.append((z, num_episodes, mean, hw))

            # update best EETDR scores if necessary
            if verbose:
                self.update_and_print(num_episodes, mean, hw, Q)
            else:
                self.update_best_scores(mean, hw, Q)

        time_elapsed = perf_counter() - time_start
        print(f"Executed {num_replications} algorithm reps in {time_elapsed:0.4f} seconds.")

        self.total_training_reps += num_replications
        self.total_episodes += num_episodes
        # Update execution time record
        total_time = (self.total_training_reps * self.avg_execution_time) + time_elapsed
        self.avg_execution_time = total_time / self.total_training_reps
