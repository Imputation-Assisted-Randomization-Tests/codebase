import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeGeneratorMissingCovariate as Generator
import MultipleOutcomeDataGenerator as GeneratorMutiple
import RandomizationTest as RandomizationTest
from sklearn.impute import SimpleImputer
import os
import lightgbm as lgb
import xgboost as xgb
import iArt

# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3
Iter = 10000



def run_simulations_for_model(model, lambda_mapping, Nsize=-1,  small_size=False):
    # Configuration based on the size of the dataset
    if small_size:
        Nsize = 50
        configs = [
            {"filepath": f"../../output/Simulation0.25Missing/HPC_power_{Nsize}_model{model}"},
        ]
    else:
        Nsize = 1000
        configs = [
            {"filepath": f"../../output/Simulation0.25Missing/HPC_power_{Nsize}_model{model}"},
        ]
        

    for coef, Missing_lambda in lambda_mapping.items():
        for config in configs:
            run_simulation(Nsize=Nsize, filepath = config['filepath'], beta_coef = coef, Missing_lambda = Missing_lambda, small_size = small_size, model = model)
    
def run_simulation(*,Nsize, filepath,beta_coef,  Missing_lambda, model = 0,  small_size = True, verbose = False):

    DataGen = Generator.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
    X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N = Nsize)
    reject, p_values= Framework.test(Z, X, M, Y,strata_size = 10, L=Iter, G = None,verbose=verbose)
    # Append p-values to corresponding lists
    values_oracle = [ *p_values, reject]
    #mask Y with M
    Y = np.ma.masked_array(Y, mask=M)
    Y = Y.filled(np.nan)

    #Median imputer
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=median_imputer,L=Iter, verbose=verbose, rankAdjust=True)
    values_median = [ *p_values, reject ]

    #linear regression imputer
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=BayesianRidge,L=Iter, verbose=verbose )
    values_LR = [ *p_values, reject ]

    if small_size == True:
        #XGBoost 
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=XGBoost,L=Iter, verbose=verbose)
        values_xgboost = [ *p_values, reject ]
    else:
        #LightGBM
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter)
        reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=LightGBM,L=Iter,verbose=verbose)
        values_lightgbm = [ *p_values, reject ]

    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)
    np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
    np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
    np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef, task_id), values_LR)
    if small_size == True:
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)
    else:
        np.save('%s/%f/p_values_lightgbm_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        save_file = True
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    # Model 1
    beta_to_lambda = {0.0: 2.159275141001102, 0.07: 2.165387531267955, 0.14: 2.285935405246937, 0.21: 2.258923945496463, 0.28: 2.2980720651301794, 0.35: 2.3679216299985613}
    run_simulations_for_model(1, beta_to_lambda, small_size=False)

    beta_to_lambda = {0.0: 2.1577587265653126, 0.25: 2.2946233479956843, 0.5: 2.42339283727788, 0.75: 2.544154767644711, 1.0: 2.669166349074493, 1.25: 2.792645016605368}
    run_simulations_for_model(1, beta_to_lambda, small_size=True)
    
    # Model 2
    beta_to_lambda = {0.0: 14.376830203817725, 0.16: 14.492781397662549, 0.32: 14.636259203432914, 0.48: 14.790662640235277, 0.64: 14.902477227186191, 0.8: 14.995429287214796}
    run_simulations_for_model(2, beta_to_lambda, small_size=False)

    beta_to_lambda = {0.0: 14.335704307321487, 0.8: 14.971101330156632, 1.6: 15.366375386649604, 2.4: 15.724510367662774, 3.2: 15.831265197313604, 4.0: 16.011150941155087}
    run_simulations_for_model(2, beta_to_lambda, small_size=True)

    # Model 3
    beta_to_lambda = {0.0: 14.43119646829717, 0.06: 14.530199258897895, 0.12: 14.74631511872901, 0.18: 14.90822250678929, 0.24: 14.947558384606348, 0.3: 14.979303880359883}
    run_simulations_for_model(3, beta_to_lambda, small_size=False)

    beta_to_lambda = {0.0: 14.390422997290685, 0.25: 14.909362702476912, 0.5: 15.219787798636258, 0.75: 15.4701421427122, 1.0: 15.625851714156521, 1.25: 15.831324938012127}
    run_simulations_for_model(3, beta_to_lambda, small_size=True)

    # Model 4
    beta_to_lambda = {0.0: 15.359698674885047, 0.06: 15.507224279021253, 0.12: 15.675599389006583, 0.18: 15.744503702370242, 0.24: 15.778177240810757, 0.3: 15.8935570369039}
    run_simulations_for_model(4, beta_to_lambda, small_size=False)

    beta_to_lambda = {0.0: 15.391438190996098, 0.25: 15.871854826261341, 0.5: 16.34293913102228, 0.75: 16.45643215605396, 1.0: 16.556851722322666, 1.25: 16.760752915013537}
    run_simulations_for_model(4, beta_to_lambda, small_size=True)
