#  Imputation-Assisted Randomization Tests

Design-based causal inference, also known as randomization-based or finite-population causal inference, is one of the most widely used causal inference frameworks, largely due to the merit that its validity can be guaranteed by the study design (e.g., randomized experiments) and does not require assuming specific outcome-generating distributions or super-population models. Despite its advantages, design-based causal inference can still suffer from other data-related issues, among which outcome missingness is a prevalent and significant challenge. This work systematically studies the outcome missingness problem in design-based causal inference. First, we propose a general and flexible outcome missingness mechanism that can facilitate finite-population-exact randomization tests for the null effect. Second, under this flexible missingness mechanism, we propose a general framework called ``imputation and re-imputation" for conducting finite-population-exact randomization tests in design-based causal inference with missing outcomes. This framework can incorporate any imputation algorithms (from linear models to advanced machine learning-based imputation algorithms) while ensuring finite-population-exact type-I error rate control. Third, we extend our framework to conduct covariate adjustment in randomization tests and construct finite-population-valid confidence regions with missing outcomes. Our framework is evaluated via extensive simulation studies and applied to a large-scale randomized experiment. Corresponding user-friendly Python and R packages are also developed.

## Repository Structure

- `code/` - Contains all scripts and code for the project.
  - `application/` - Application specific scripts[(Detail)](./code/Application/README.md).
  - `simulation/` - Simulation scripts, tools[(Detail)](./code/Simulation/README.md).
- `data/` - Raw data files used in analyses, script for data preprocessing[(Detail)](./data/README.md).
- `manuscript/` - Source files for the manuscript, including LaTeX or Markdown files.
- `output/` - Results and outputs from computations[(Detail)](./output/README.md).

## Setting Up Conda Environment

Navigate to the main directory of the codebase to start this.

```bash
# Step 1: Create the Python Virtual Environment
python -m venv iart_env

# Step 2: Activate the Environment
# For Unix or MacOS:
source iart_env/bin/activate

# For Windows:
.\iart_env\Scripts\activate

# Step 3: Upgrade pip (optional but recommended)
pip install --upgrade pip

# Step 4: Install Requirements Using pip
pip install -r requirements.txt
```

