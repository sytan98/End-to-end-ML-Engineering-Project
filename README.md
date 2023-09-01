# End-to-end-ML-Engineering-Project

This project implements an entire ML pipeline for Singapore public housing resale price prediction. 
The ML models were inspired from this kaggle article: https://www.kaggle.com/code/lizexi/singapore-s-public-housing-eda-lasso-catboost#visualize-predictions

It uses MLFlow to manage model experimentation, tracking and serving. 
Airflow is used to schedule the whole pipeline to enable retraining overtime as new data becomes available and to account for data distribution shifts.

## Architecture

## TODO
- Data versioning: Point in time records for features
- Model validation: Check if new model really outperforms current model before deployment
- Testing