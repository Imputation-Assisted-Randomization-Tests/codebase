import numpy as np
import os
import re  # Import the regex module
import matplotlib.pyplot as plt

results_dir = 'ConfidenceSets'  # Directory containing the results

import os
os.chdir('../../Output')

files = os.listdir(results_dir)

# Initialize a dictionary to store results by imputer type
results_by_imputer = {}

# Define a regex pattern to match the imputer type more accurately.
# This pattern assumes that the imputer type is followed by an underscore and possibly a negative sign before the numeric value.
pattern = re.compile(r'test_([a-zA-Z]+)_?-?\d')

# Loop through each file, load the result, and organize by imputer type
for file in files:
    if file.startswith('test_') and file.endswith('.npy'):
        match = pattern.search(file)  # Search for the pattern in the filename
        if match:
            imputer_type = match.group(1)  # Extract imputer type from the regex match
            result = np.load(os.path.join(results_dir, file))
            
            # If the imputer type is not in the dictionary, add it with an empty list
            if imputer_type not in results_by_imputer:
                results_by_imputer[imputer_type] = []
            
            # Append the result to the correct list based on imputer type
            results_by_imputer[imputer_type].append(result)

# Process each imputer type to calculate confidence sets or other statistics
for imputer_type, results in results_by_imputer.items():
    results = np.array(results)  # Convert list to numpy array for easier processing

    # Assuming the last element in the result is the p-value
    # We consider the null hypothesis rejected if p-value <= 0.1
    confidence_set = results[results[:, 2] > 0.05, 0]

    # Sort the confidence set
    confidence_set = np.sort(confidence_set)
    
    # print and write the confidence set to a file
    print(f"One-way confidence set for beta using {imputer_type} imputer:", confidence_set)
    with open('confidence_sets.txt', 'a') as file:
        file.write(f"One-way confidence set for beta using {imputer_type} imputer:\n")
        file.write(np.array2string(confidence_set, separator=', '))
        file.write('\n\n')

