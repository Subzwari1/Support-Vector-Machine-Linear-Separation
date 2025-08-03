# Support-Vector-Machine-Linear-Separation
Support Vector Machine (SVM) Classification with Bilevel Cross-Validation in Octave

This project implements a **Support Vector Machine (SVM)** classifier using Octave to separate two classes of points from a given dataset. The implementation strictly follows the given project requirements, including dual optimization, primal recovery, and cross-validation for model selection and evaluation.

Key Features
- Dataset: Loads a .mat dataset containing feature matrix X and labels y.

- Preprocessing:
  - Standardizes features to zero mean and unit variance.
  - Ensures labels are mapped to +1 and -1.
 
- Model:
  - Implements linear kernel SVM using dual optimization formulation.
  - Recovers primal variables (weight vector v and bias γ) from dual solution.
 
- Validation:
  - Performs bilevel cross-validation:
  - Outer 10-fold CV for performance evaluation.

- Hyperparameter Tuning:
  - Candidate C values generated using logspace(-2, 2, 5).

- Performance Metrics:
  - Training and Testing:
    - Accuracy (Correctness)
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1-Score

- Visualization:
  - Plots decision boundary, margins (H+, H−), and data points for each outer fold when features are 2D.
  - Saves plots as images for analysis.
  
Technologies Used
  - Octave (GNU Octave GUI, version ≥ 6.0)
  - Optim and Statistics packages for optimization and CV partitioning.

How It Works
1. Load Dataset (dataset29.mat): Feature matrix X and label vector y.
2. Standardize Data and remap labels to ±1.
3. Perform Cross-Validation:

 - Outer loop: Split data into 10 folds (train/test).
 - Inner loop: Split training data into 5 folds for model selection.

4. Solve Dual SVM Problem:
 - Optimize Lagrangian dual using quadratic programming (qp).
 - Compute primal weight vector v and bias γ.

5. Evaluate Model:
  - Compute metrics on training and test sets.
  - Average metrics across folds.

6. Plot Decision Boundaries:
  - Hyperplane H(v, γ), margins H+, and H−.
