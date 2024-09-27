# Kaggle House Prices - Predicting Sale Prices

This repository contains the work done for the **Kaggle House Prices** competition. The goal is to predict the sale prices of homes based on various features provided in the dataset. We explore different models and ultimately use **GradientBoostingRegressor** for final predictions.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
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

## Installation

To run the code in this repository, you'll need to have **Python 3.x** installed. The project also requires several dependencies, which are listed in the `requirements.txt` file. You can install them by running:

```bash
pip install -r requirements.txt
