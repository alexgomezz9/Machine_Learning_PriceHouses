# Kaggle House Prices - Predicting Sale Prices

This repository contains the work done for the **Kaggle House Prices** competition. The goal is to predict the sale prices of homes based on various features provided in the dataset. We explore different models and ultimately use **GradientBoostingRegressor** for final predictions.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Models Used](#models-used)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict house sale prices using a dataset from Kaggle. The dataset contains a variety of features that describe the homes, such as size, neighborhood, year built, and more. We experimented with different models, including **Random Forest** and **Gradient Boosting**, before finalizing our solution.

### Key steps:
- Data cleaning and preprocessing.
- Feature engineering and handling missing data.
- Model selection: various regression models were tested.
- Hyperparameter tuning using **RandomizedSearchCV**.
- Final model trained using **GradientBoostingRegressor**.

## Data

The dataset used in this project can be found on Kaggle under the **House Prices** competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

The dataset includes:
- **train.csv**: Training data with features and the target (`SalePrice`).
- **test.csv**: Test data without labels.

## Models Used

We tested several machine learning models to predict house prices, including:

1. **Random Forest Regressor**: An ensemble model that creates multiple decision trees and averages their predictions.
2. **Gradient Boosting Regressor** (Final Model): A boosting technique that sequentially improves the model by minimizing errors from previous trees.

### Hyperparameter Tuning

We used **RandomizedSearchCV** to fine-tune hyperparameters for the best performance on the validation set. Some of the tuned parameters include:
- `n_estimators`
- `learning_rate`
- `max_depth`

## Results

The final model used is **GradientBoostingRegressor**, which provided the lowest Mean Absolute Error (MAE) during validation.

Final **MAE** on validation set: **18,143.07**

## Usage

To train the model and generate predictions for the test set, follow these steps:

1. Clone this repository by copying the URL of your repository and using the git clone command.

2. Run the data preprocessing and model training script by executing the appropriate Python files for preprocessing and training.

The predictions will be saved in a file named `predicciones_submission.csv`.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
