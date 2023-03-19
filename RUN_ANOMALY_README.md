# How to run project

Project source files are stored in folder `anomaly/`.

0. Before run project you should download Kaggle API key if you don't have one. This API key is needed to download
dataset from `kaggle.com`.
Go to `kaggle.com`, open "Account" tab and click "Create new API Token". Then `kaggle.json` file is downloaded. Then you 
should move this `kaggle.json` file to folder `~/.kaggle/`

1. To run the project use this command: `dvc repro`

2. Check scores for train and test datasets: `dvc metrics show`

3. Options
   - Stacking classifier with 3 estimators (CatBoost, XGB, LGBM) is used by default. You can change it with parameter
   `--clf_model` for `python anomaly/predict.py` command (`predict` stage in `dvc.yaml`). Options: `catboost`, `logreg`,
   `xgboost`, `staking`. Selected classifier is loaded from pickle file. 
   - Classifier is loaded from a pickle file by default. You can train classifier with parameters stored
   in json files in `anomaly/models_params` folder.
   Just use parameter `--fit_clf True` for `python anomaly/predict.py` command (`predict` stage in `dvc.yaml`).
   And use parameter `--clf_model` for classifier selection. Possible options: `logreg`, `svc_lin`, `svc_rbf`, `catboost`,
   `xgboost`, `lightgbm`, `staking`. Be careful. Some models training takes too much time (svc, busting, stacking).
   Stacking takes a few hours on CPU.
   - By default, pipeline for data preprocessing (`prep_data_pipe`) is loaded from a pickle file. You can fit this
   pipeline on the train dataset. Use option `--fit_prep_data_pipe True` for `python anomaly/predict.py` command (`predict` 
   stage in `dvc.yaml`).