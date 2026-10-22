# Bank Telemarketing Campaign Optimization

## Project Overview
A banking institution requires optimization of its telemarketing resources to maximize term deposit conversions. This repository contains a complete data science workflow using Python to predict campaign outcomes and segment customers. The analysis identifies high-value customer profiles, evaluates predictive models, and quantifies the financial return of targeted marketing. 

## Data Preparation
The dataset originates from a Portuguese banking institution. The initial processing step removes the `duration` variable. The length of a phone call is only known after the interaction concludes. Including this variable creates data leakage by producing an artificially perfect model that cannot function in a real deployment. The data is partitioned into a sixty percent training set and a forty percent validation set. Stratified sampling preserves the eleven percent conversion rate across both partitions.

## Customer Segmentation
K-Means clustering is applied to standardized numerical features to identify distinct customer behaviors. The elbow method determines that three clusters provide the optimal balance between complexity and within-cluster variance. Analysis of these segments reveals a high-value customer group with a twenty-two percent baseline conversion rate. This specific segment justifies a specialized marketing campaign. 

## Predictive Modeling
Two supervised learning models are evaluated: Logistic Regression for interpretability and Random Forest for capturing non-linear relationships. The rare nature of the positive class requires specific treatment. Synthetic Minority Over-sampling Technique balances the classes strictly within the ten-fold cross-validation process. This pipeline architecture prevents validation data from leaking into the training process, ensuring an honest evaluation of the model's predictive power.

## Model Evaluation
The Random Forest model achieves a validation Area Under the Curve of 0.7405. At the default 0.5 probability threshold, the model attains forty percent sensitivity, identifying forty percent of all likely buyers. A decile lift chart demonstrates the business value. Targeting the top ten percent of the model's ranked probabilities increases the response rate from the baseline eleven percent to approximately fifty percent. This four-fold return allows the bank to reduce call volume by ninety percent while capturing the majority of potential sales. Variable importance analysis identifies bank balance, age, and previous campaign success as the primary drivers of conversion.

## Repository Structure
`Bank_Marketing_Analysis.ipynb` contains the complete executable code, data preprocessing, model training, and visualizations. The notebook includes the Elbow plot, ROC curve, Lift chart, and Variable Importance plot.

## Requirements
The project requires Python 3.8 or higher. Required libraries include pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, and seaborn. The dataset file `bank.csv` must be placed in the root directory before running the notebook.
