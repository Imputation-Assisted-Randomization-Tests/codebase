# We will run the iArt test on the cleaned dataset using different imputation methods
import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import iArt

# Load the arrays from the .npz file
arrays = np.load('../../Data/CleanedDataset.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']

# Run the iArt test
file_path = "../../Output/p_values.txt"
L = 10000
verbose = 1
random_state = 0
threshholdForX = 0.0

# Write the results with one-sided test
with open(file_path, 'a') as file:
    file.write("imputation-assisted randomization test\n")
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= 'median', verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Method 1 (Median Imputation): " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose = verbose,mode = 'cluster',threshholdForX = threshholdForX,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Method 2 (Algo 1 - Linear): " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose=verbose,mode = 'cluster', threshholdForX = threshholdForX,covariate_adjustment=1,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Method 3 (Algo 1 - Linear): " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y,G='lightgbm',S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Method 4 (Algo 1 - Boosting): " + str(result) + '\n')

    result = iArt.test(Z=Z, X=X, Y=Y,G='lightgbm',S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster', covariate_adjustment=3,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Method 5 (Algo 2 - Boosting): " + str(result) + '\n')