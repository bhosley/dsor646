from MDP import MDP_Tiled

import numpy as np
from time import perf_counter
from tiles3 import IHT

class LambdaSARSA(MDP_Tiled):
    name = "SARSA (lambda) with linear tile coding VFA scheme"

    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.algorithm_name = self.name

        """ Tunable Hyperparameters """
        self.lam_a      = kwargs.get('eps_a',0.8)   # Trace decay parameter
        self.lam_b      = kwargs.get('eps_b',0.2)   # Trace decay parameter
        self.Δ_clip     = kwargs.get('clip',10)     # TD error clip (for stability)


    def lam(self,n) :
        return self.lam_a/(1+n)**self.lam_b

    def train(self, num_replications, num_episodes, verbose=False):
        time_start = perf_counter()
        for z in range(num_replications):
            # theta vector of basis function weights
            w = self.qinit*np.ones((self.max_size,1))/self.num_tiles
            zeta = np.zeros((self.max_size,1))      # eligibility trace vector
            Nsa  = np.zeros((self.max_size,1))      # state action counter
            iht_VFA = IHT(self.max_size)            # integer hash table

            print(f"\nSARSA(alpha_a={self.alpha_a: <3.2f},alpha_b={self.alpha_b:<3.2f}," 
                 +f"eps_a={self.eps_a:<3.2f},eps_b={self.eps_b:<3.2f}, lam_a rep{z}...")

            for m in range(num_episodes): # M
                terminated,truncated = False,False                  # Episode complete flags
                Gm = 0                                              # Episode reward
                np.random.seed(int(m + 1e6*z + 1e5*self.offset))    # Set rng seed
                state = self.env.reset(seed=int(z + 1e6 + m + 1e5*self.offset))[0]

                # select action based on epsilon-greedy exploration mechanism
                action = self.get_action(state,(w,iht_VFA),self.epsilon(m))

                # SARSA main loop (first nm1 transitions until end of episode)
                while not(terminated or truncated):
                    # Apply action and observe system information
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    Gm += reward    # Update episode cumulative reward

                    # select action based on epsilon-greedy exploration mechanism
                    next_action = self.get_action(state,(w,iht_VFA),self.epsilon(m))

                    # Compute qhat and TD error
                    qhat = reward + (1-terminated)*self.gamma*self.Qbar(next_state,next_action,w,iht_VFA)
                    Δ = np.clip(-self.Δ_clip, qhat - self.Qbar(state,action,w,iht_VFA),self.Δ_clip)

                    # Update:
                    zeta = self.gamma*self.lam(m)*zeta
                    active_tiles = self.gradQbar(state,action,iht_VFA)
                    zeta[active_tiles] += 1         # Eligibility trace
                    Nsa[active_tiles] += 1          # State-action counter
                    w += self.alpha(Nsa)*Δ*zeta     # w vector
                    state = np.copy(next_state)     # State
                    action = next_action            # Action - np.copy creates an array -> Error

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


class LambdaSARSA_boltzmann(LambdaSARSA):
    name = "SARSA (lambda) with linear tile coding VFA scheme and Boltzmann Exploration"

    def __init__(self, env, n=1, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.algorithm_name = self.name
        self.tau            = kwargs.get('tau',2.0)
        self._tau_          = self.tau
        self.cooling_rate   = kwargs.get('cooling_rate',0.001)


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