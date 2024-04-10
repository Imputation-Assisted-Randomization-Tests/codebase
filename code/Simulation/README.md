# Simulation Guide

This document provides instructions on how to run the Python scripts in the `YourLocationOfTheFolder/Code/Simulation` directory for the simulation.

## Directory Structure

- `CalculatePowerForMultipleOutcome.py`: Main script to run the analysis and compute p-values for multiple outcomes.
- `CalculatePowerForSingleOutcome.py`: Main script to run the analysis and compute p-values for single outcomes.
- `iArt.py`: Imputation-Assisted Randomization Tests (iArt).
- `MultipleOutcomeModelGenerator.py`: Generates Model 5 for multiple outcomes.
- `RandomizationTest.py`: Script for conducting Oracle randomization tests.
- `SingleOutcomeModelGenerator.py`: Generates Models 1, 2, 3, 4, and 6 for single outcomes.


## Running the Scripts

### Main Analysis

To run the main analysis for multiple outcomes or single outcomes, navigate to the current folder and execute the following command:
```
python CalculatePowerForSingleOutcome.py <job_array_index>
python CalculatePowerForMultipleOutcome.py <job_array_index>
```
This will generate a folder `Simulation` in the `Output` folder. Here, p-values for various methods including Oracle, Median Imputation, Algo 1 (Linear), Algo 1 (Boosting), Algo 2 (Linear), and Algo 2 (Boosting) for different models will be stored under separate folders for different models.


#### Output Files

Results will be saved in the `Output/Simulation` directory. Each model, except Model 5, will have six subdirectories HPC_power_50_model`i`, HPC_power_50_model`i`_adjusted_Xgboost, HPC_power_50_model`i`_adjusted_LR, HPC_power_1000_model`i`, HPC_power_1000_model`i`_adjusted_LightGBM, and HPC_power_1000_model`i`_adjusted_LR representing the method of covariate adjustment. Within each subdirectory, folders for different beta values contain the final p-values. Each p-value file includes:

- **Reject**: This boolean indicates whether the null hypothesis for the beta value was rejected.
- **P-Value**: This is the probability of observing the given result, or something more extreme, under the null hypothesis.


## Note on Performance

The scripts are designed to be run in parallel on a high-performance computing cluster. The main analysis scripts, in particular, are parallelized across 2000 cores. Running these scripts without such parallelization will result in significantly longer execution times.
