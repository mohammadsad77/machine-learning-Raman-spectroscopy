# Spectral Data Classification Project

This repository contains three Python scripts for processing spectral data, training a machine learning model,
and evaluating its performance. The scripts are designed to work with a dataset stored in `C:/Users/PayaPc.Com/Downloads/43591_2023_57_MOESM2_ESM/`,
which includes anonymized spectral data for classifying materials into 7 classes: '0', 'PE', 'PET', 'PP', 'PS', 'PVC', and 'CuPC'.

## Scripts Overview

### 1. Step 1: Data Preprocessing and Splitting 
#### Purpose
- Loads spectral data from JSON files in `DataAnonym/`.
- Extracts features by computing the gradient, slicing to 611 elements (from index 281 to 892), and normalizing with `StandardScaler`.
- Splits the data into training (70%), validation (20%), and test (10%) sets using stratified sampling.
- Saves the split datasets as `.npz` files in `DataSplit/`.
- Plots one raw and one processed spectrum for each class in separate figures.

#### Key Steps
- **Feature Extraction:** Computes the gradient of raw spectral data, slices it, and normalizes it.
- **Data Splitting:** Uses `train_test_split` with `stratify` to maintain class distribution.
- **Plotting:** Displays raw (full-length) and processed (611-element) spectra for one sample per class.

#### Running the Script
```bash
python step1Mohammad.py
#### Output
.npz files: train.npz, val.npz, test.npz in DataSplit/.

Separate plots for raw and processed spectra for each class (up to 14 figures).
###############################################
2. Step 2: Model Training with Stacking 
Purpose
Trains a stacking classifier using multiple base models (MLP, SGD, RandomForest, SVC) and a LogisticRegression meta-model.

Applies class weights to handle imbalance (e.g., class '0' dominance).

Evaluates the model on training, validation, and test sets.

Saves the trained model as a .pkl file.

Key Steps
Loading Data: Loads pre-split datasets from DataSplit/.

Class Weights: Computes weights based on class distribution to balance training.

Model Definition: Uses StackingClassifier with 4 base models and LogisticRegression as the final estimator.

Evaluation: Calculates accuracy on training, validation, and test sets.

Running the Script
python step2Mohammad.py
Output
Printed accuracies for training, validation, and test sets.

Saved model: stacking_model_scikit.pkl in DataModels/.

###############################################
3. Step 3: Model Evaluation and Visualization 
Purpose
Loads the trained stacking model and test data.

Evaluates the model’s performance per class using Recall, Precision, Accuracy, Tested Instances Ratio, and AUC.

Visualizes these metrics in separate bar charts for each class (0 to 6).

Key Steps
Loading: Loads the model (stacking_model_scikit.pkl) and test data (test.npz).

Evaluation: Computes multiple metrics for each class using binary classification per class.

Plotting: Creates 5 bar charts (one per metric) with values displayed above each bar.

Running the Script

python script3.py
Output
Printed metrics (Recall, Precision, Accuracy, Tested Instances Ratio, AUC) for each class.

Five bar charts visualizing the metrics across all classes.

########################################################

