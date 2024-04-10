# Application Guide

This document provides instructions on how to run the Python scripts in the `YourLocationOfTheFolder/Code/Application` directory for the WFHS dataset analysis.

## Directory Structure

- `Main.py`: Main script to run the analysis to get the pvalues.
- `iArt.py`: Imputation-Assisted Randomization Tests (iArt).
- `ConfidenceSets.py`: Computes confidence sets.
- `AggregateResultsForConfidenceSets.py`: Aggregates confidence sets results.
- `Output/`: Directory where all outputs will be saved.

## Running the Scripts

### Main Analysis

To run the main analysis, navigate to the current folder and execute the following command:
```
python Main.py
```
This will generate a file `p_values.txt` in the `Output` folder. It will contain all the p-values for Median Imputation, Algo 1 (Linear), Algo 1 (Boosting), Algo 2 (Linear), and Algo 2 (Boosting) for the WFHS dataset.


### Confidence Sets

To compute confidence sets, you need to run the `ConfidenceSets.py` script. This should be executed within a SLURM managed cluster environment, using a job array index to specify the particular job for which you wish to compute the confidence set.

```bash
python ConfidenceSets.py <job_array_index>
```

The `<job_array_index>` must be an integer that corresponds to the specific job index within the cluster's job array. If you do not include this argument, the script will not run and will return an error.

#### Output Files

The results from this script will be saved in the `Output/ConfidenceSets` directory. Each filename, such as `test_lightgbmcovariateadjustment_-0.002501250625313034.npy`, indicates the test result for a LightGBM model with the corresponding beta value. The file contains:

- **Beta**: This is the estimated value of the parameter from the model.
- **Reject**: This boolean indicates whether the null hypothesis for the beta value was rejected.
- **P-Value**: This is the probability of observing the given result, or something more extreme, under the null hypothesis.

For example, to run the script for the job array index 900, you would use the command:

```bash
python ConfidenceSets.py 1000
```

This command processes the 1000th job and generates the output files, which will include the beta value, the rejection status, and the p-value necessary for your statistical analysis.


### Aggregating Confidence Sets Results

To aggregate the confidence Sets results and generate `confidencesets.txt`, use:
```
python AggregateResultsForConfidenceSets.py
```
The `confidence_sets.txt` file will be created in the `Output` folder.

## Note on Performance

The scripts are designed to be run in parallel on a high-performance computing cluster. The confidence sets calculations, in particular, are parallelized across 2000 cores. Running these scripts without such parallelization will result in significantly longer execution times.
