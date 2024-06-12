import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats.qmc import LatinHypercube
from joblib import Parallel, delayed
from datetime import datetime
from time import perf_counter
from statsmodels.formula.api import ols
import statsmodels.api as sm

class LHS_Experiment():
    def __init__(self, algorithm, env, features, **kwargs) -> None:
        self.algorithm = algorithm
        self.env = env
        self.features = features
        # ["alpha_a", "alpha_b", "eps_a", "eps_b"]
        self.column_names = ["Run Index"] + features + ["Sup EETDR", "Sup EETDR hw", 
                             "Mean Max EETDR", "Mean Max EETDR hw", "Time-Avg EETDR", 
                             "Time-Avg EETDR hw", "Secs per run", "Score"]
        self.NUM_CPU_PROCS  = kwargs.get('num_cpu_procs',16)     # Number of CPU threads to use
        self.num_episodes   = kwargs.get('episodes',int(1e3))
        self.replications   = kwargs.get('replications',10)
        self.runs           = kwargs.get('runs',50)
        self.verbose        = kwargs.get('verbose',False)

        self.results_table = None
        self.factor_table = None


    def single_experiment(self, run_index:int, factors:np.ndarray) -> tuple[int,float,float,float]:
        run_start_time = perf_counter()
        #algo = self.algorithm(*factors)
        algo = self.algorithm(self.env, **dict(zip(self.features,factors)))
        algo.train(self.replications, self.num_episodes)
        maxETDR, maxETDRhw, meanMaxTestEETDR, maxTestHW, meanAULC, hwAULC, time =algo.get_results()
        alg_score = 0.6*(meanMaxTestEETDR-maxTestHW) + 0.4*(meanAULC-hwAULC)
        if self.verbose: print(f"Complete experiment run {run_index} with a score of "
                               + f"{alg_score:.2f} ({perf_counter() - run_start_time:.1f}s)")
        return run_index, maxETDR, maxETDRhw, meanMaxTestEETDR, maxTestHW, meanAULC, hwAULC, time

    def parallel_lhs(self, rng_seed=0):
        """ Execute LHS Experiment in Parallel """
        sampler = LatinHypercube(len(self.features), scramble=False, 
                                 optimization="lloyd", seed=rng_seed)
        factor_table = sampler.random(n=self.runs)
        if self.verbose: print(f"\nInitializing LHS experiment with {self.runs} runs...")
        start_time = perf_counter()
        parallel_manager = Parallel(n_jobs = self.NUM_CPU_PROCS)
        run_list = (delayed(self.single_experiment)(run_index, factor_table[run_index]) 
                    for run_index in range (self.runs) )
        #execute the list of run_experiment() calls in parallel
        if self.verbose: print ("\nExecuting experiment...")
        results_table = parallel_manager(run_list)
        results_table = np.array(results_table)
        if self.verbose: print (f"\n\nCompleted experiment ({perf_counter() - start_time:.3f}s)")

        # Combine the factor and results tables, add column headers, and save the data to a CSV.
        # Compute algorithm run score, the avg 95% CI lower bound for max and mean performance.
        maxEETDR_95CI_LB = results_table[:,3] - results_table[:,4]
        meanEETDR_95CI_LB = results_table[:,5] - results_table[:,6]
        score = 0.6*maxEETDR_95CI_LB + 0.4*meanEETDR_95CI_LB
        results_table = np.column_stack((results_table[:,0], factor_table, 
                                         results_table[:,1:], score))
        # save data for performance scatter plot
        self.results_table = np.row_stack((self.column_names, results_table))
        self.factor_table = factor_table

    def export_results(self):
        filename_DOE = (f"{self.algorithm.name}_results_DOE_" 
                        + datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv")
        np.savetxt(filename_DOE, self.results_table, delimiter = ",", fmt = "%s")

    def anova(self):
        # Input data
        X = self.factor_table

        # Generate full factorial polynomial function up to degree 2
        poly = PolynomialFeatures(2)
        X_poly = poly.fit_transform(X)

        # Clean up the feature names
        input_features = self.column_names[1: 1 + len(self.features)] # Run index and features
        feature_names = [name.replace(' ','_').replace('^','_pow_').replace('*','_times_')
                        for name in poly.get_feature_names_out(input_features=input_features)]
        df = pd.DataFrame(X_poly, columns=feature_names)

        # define response variable
        max_ind = self.column_names.index("Sup EETDR")
        mean_ind = self.column_names.index("Mean Max EETDR")
        maxEETDR_95CI_LB  = (np.array(self.results_table[1:,max_ind],float) 
                           - np.array(self.results_table[1:,max_ind+1],float))
        meanEETDR_95CI_LB = (np.array(self.results_table[1:,mean_ind],float) 
                           - np.array(self.results_table[1:,mean_ind+1],float))
        score = 0.6*maxEETDR_95CI_LB + 0.4*meanEETDR_95CI_LB
        df['AlgScore'] = score

        # Create the formula string for the OLS model
        # Exclude the first column (the constant term) from the predictors
        predictors = '+'.join(df.columns[1:-1]) # Exclude '1' and 'y'
        formula = f'AlgScore ~ {predictors}'

        # Create and fit the OLS model
        model = ols(formula, data=df)
        results = model.fit()

        # Display the summary
        print ("\n\n" + results.summary())

        # Perform ANOVA and display the table
        anova_results = sm.stats.anova_lm(results, typ=2)
        print(anova_results)

    def plot_results(self,x_var = "Mean Max EETDR", y_var = "Time-Avg EETDR"):
        x_ind = self.column_names.index(x_var)
        y_ind = self.column_names.index(y_var)
        x = np.array(self.results_table[1:,x_ind],float) # 0-index appears to be title
        y = np.array(self.results_table[1:,y_ind],float) # 0-index appears to be title
        # create scatter plot
        plt. scatter(x, y, label=f"{self.algorithm.name} -- {self.replications} reps per run, "
                     + f"{self.num_episodes} episodes per rep")
        # setting title and labels
        plt.title(f"{self.algorithm.name} LHS DOE Performance Results")
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.grid()                                  # grid on
        plt.legend(loc='upper left', fontsize=7)    # legend on
        plt.show()                                  # display the plot

    def plot_param_comparison(self):
        y = np.array(self.results_table[1:,-1],float) # 0-index appears to be title
        # create scatter plot
        for i,param in enumerate(self.features):
            plt.scatter(np.array(self.results_table[1:,i+1],float), y , label=param)
        # setting title and labels
        plt.title(f"{self.algorithm.name} LHS DOE Performance Results -- "
                     + f"{self.replications} reps per run, {self.num_episodes} episodes per rep")
        plt.ylabel("Score")
        plt.grid()                                  # grid on
        plt.legend(loc='upper left', fontsize=7)    # legend on
        plt.show()                                  # display the plot