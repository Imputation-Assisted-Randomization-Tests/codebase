# WFHS Dataset Preprocessing Guide

This guide provides instructions on how to preprocess the WFHS dataset to extract treatment, covariates, outcome, and cluster indicators using the script `DataPreprocessing.py`.

## Data Directory Structure

The `Data` directory contains the following relevant files:

- `36158-0001-Data.tsv`: The raw dataset file.
- `36158-Codebook.pdf`: The codebook for the dataset explaining the variables.
- `36158-User_guide.pdf`: The user guide for the dataset.
- `DataPreprocessing.py`: The Python script for preprocessing the dataset.
- `CleanedDataset.npz`: The file where the cleaned data will be saved.
- `README.md`: The readme file containing general information.

## Preprocessing the Data

To preprocess the data, follow these steps:

1. Navigate to the `Data` directory where the `DataPreprocessing.py` script is located.

2. Run the script using the following command in the terminal:
```bash
python ConfidenceSets.py <job_array_index>
```
4. 
The script will process the raw data and extract the following components:

- `Z`: Treatment variable.
- `X`: Covariates.
- `Y`: Outcome variable.
- `S`: Cluster number.

4. Upon successful execution, the script will save these components into `CleanedDataset.npz` in the NumPy array format within the `Data` directory.

Make sure to refer to the `36158-Codebook.pdf` and `36158-User_guide.pdf` for detailed information on the variables and the structure of the dataset.

## Next Steps

After preprocessing, you can proceed with the analysis as per the instructions in the `README.md` in Application folder.


