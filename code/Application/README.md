# Application Guide

This document provides instructions on how to run the Python scripts in the `YourLocationOfTheFolder/code/application` directory for the WFHS dataset analysis.

## Directory Structure

- `Main.py`: Main script to run the analysis to get the pvalues.
- `iArt.py`: Imputation-Assisted Randomization Tests (iArt).
- `ConfidenceRegions.py`: Computes confidence regions.
- `AggregateResultsForConfidenceRegions.py`: Aggregates confidence regions results.
- `output/`: Directory where all outputs will be saved.

## Running the Scripts

### Main Analysis

To run the main analysis, navigate to the current folder and execute the following command:
```
python Main.py
```
This will generate a file `p_values.txt` in the `output` folder. It will contain all the p-values for Median Imputation, Algo 1 (Linear), Algo 1 (Boosting), Algo 2 (Linear), and Algo 2 (Boosting) for the WFHS dataset.


### Confidence Regions

To compute confidence regions, you need to run the `ConfidenceRegions.py` script. This should be executed within a SLURM managed cluster environment, using a job array index to specify the particular job for which you wish to compute the confidence set.

```bash
python ConfidenceRegions.py <job_array_index>
```

The `<job_array_index>` must be an integer that corresponds to the specific job index within the cluster's job array. If you do not include this argument, the script will not run and will return an error.

#### Output Files

The results from this script will be saved in the `output/ConfidenceRegions` directory. Each filename, such as `test_lightgbmcovariateadjustment_-0.002501250625313034.npy`, indicates the test result for a LightGBM model with the corresponding beta value. The file contains:

- **Beta**: This is the estimated value of the parameter from the model.
- **Reject**: This boolean indicates whether the null hypothesis for the beta value was rejected.
- **P-Value**: This is the probability of observing the given result, or something more extreme, under the null hypothesis.

For example, to run the script for the job array index 900, you would use the command:

```bash
python ConfidenceRegions.py 1000
```

This command processes the 1000th job and generates the output files, which will include the beta value, the rejection status, and the p-value necessary for your statistical analysis.


### Aggregating Confidence Regions Results

To aggregate the confidence regions results and generate `confidence_regions.txt`, use:
```
python AggregateResultsForConfidenceRegions.py
```
The `confidence_regions.txt` file will be created in the `Output` folder.

## Note on Performance

The scripts are designed to be run in parallel on a high-performance computing cluster. The confidence regions calculations, in particular, are parallelized across 1001 cores. Running these scripts without such parallelization will result in significantly longer execution times.
