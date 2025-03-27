# Simulation Guide

This guide provides detailed instructions on running Python scripts located in the `YourLocationOfTheFolder/Code/Simulation` directory.

---

## Directory Structure

Below is an overview of the primary scripts included in this directory:

| Script Name                                        | Description                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| **RunSimulation.py**                               | Main script for simulation and computation of p-values.                   |
| **iArt.py**                                        | Imputation-Assisted Randomization Tests (iArt).                           |
| **iArt_survival.py**                               | Imputation-Assisted Randomization Tests for survival analysis.            |
| **MultipleOutcomeDataGenerator.py**                | Generates simulation data for Model 5 (multiple outcomes).                |
| **RandomizationTest.py**                           | Conducts Oracle randomization tests.                                      |
| **SingleOutcomeDataGenerator.py**                  | Generates simulation data for Models 1, 2, 3, 4, and 6 (single outcomes). |
| **SurvivalDataGenerator.py** *(Appendix)*          | Generates survival analysis simulation data.                              |
| **RunSimulationSurvival.py** *(Appendix)*          | Executes survival analysis simulations.                                   |
| **RunSimulationMissingCovariate.py** *(Appendix)*  | Runs simulations involving missing covariates.                            |
| **RunSimulationModelBasedImputation.py** *(Appendix)* | Runs simulations for model-based imputation type-I error rate calculations. |
| **RunSimulationTestMissing.py** *(Appendix)*       | Executes missingness test simulations.                                    |
| **SingleOutcomeDataGeneratorMissingCovariate.py** *(Appendix)* | Generates simulation data for Models 1–4 with missing covariates.         |
| **TestMissing.py** *(Appendix)*                    | Performs missingness testing.                                             |

---

## Running the Scripts

### Main Analysis

Navigate to the simulation directory and execute scripts prefixed with `RunSimulation` as follows:

```bash
python RunSimulation.py <job_array_index>
python RunSimulationMissingCovariate.py <job_array_index>
python RunSimulationSurvival.py <job_array_index>
python RunSimulationModelBasedImputation.py <job_array_index>
python RunSimulationTestMissing.py <job_array_index>
```

This will generate a folder in the `output` folder. For example, RunSimulation.py will give you p-values for various methods, including Oracle, Median Imputation, Algo 1 (Linear), Algo 1 (Boosting), Algo 2 (Linear), and Algo 2 (Boosting) for different models will be stored under separate folders for different models.


### Output Files

Results will be saved in the `output/` directory. Each model, will have six subdirectories HPC_power_50_model`i`, HPC_power_50_model`i`_adjusted_Xgboost, HPC_power_50_model`i`_adjusted_LR, HPC_power_1000_model`i`, HPC_power_1000_model`i`_adjusted_LightGBM, and HPC_power_1000_model`i`_adjusted_LR, HPC_power_1000_model`i`_adjusted_Median representing the method of covariate adjustment. Within each subdirectory, folders for different beta values contain the final p-values. Each p-value file includes:

- **Reject**: This boolean indicates whether the null hypothesis for the beta value was rejected.
- **P-Value**: This is the probability of observing the given result, or something more extreme, under the null hypothesis.


## Note on Performance

The scripts are designed to be run in parallel on a high-performance computing cluster. The main analysis scripts, in particular, are parallelized across 2000 cores. Running these scripts without such parallelization will result in significantly longer execution times.

