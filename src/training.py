"""This code base is adopted from the below notebook
https://www.kaggle.com/code/edwingeevarughese/internet-service-churn-analysis
'''"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0301,W0108,C0103,E1101,E1137,E1136
import time
import argparse
import logging
import warnings
from utils import data_preparation, savemodel

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def hyperparams_tuning(trainmodel):
    """Defintion of Hyperparameter tuning methods"""
    best_fit_model = None
    if trainmodel == "random_forest_classifier":
        rf_model = RandomForestClassifier()
        params = {
            'n_estimators': [100, 150],
            'max_leaf_nodes': [15, 30, 45],
            'max_depth': [None, 4, 5]
            }

        grid_search = GridSearchCV(rf_model, param_grid=params, cv=3, verbose=10, n_jobs=-1)
        hyp_start_time = time.time()  # timing starts from this point for "hyp_start_time" variable
        model_rfc = grid_search.fit(X_train, y_train)
        logger.info('[RandomForestClassifier] Training Time with '
                    'Hyper parameter Tuning: %f secs', time.time()-hyp_start_time)
        logger.info('[RandomForestClassifier] Best parameters %s', model_rfc.best_params_)
        logger.info('[RandomForestClassifier] Best Accuracy score %s', model_rfc.best_score_)

        # Tuned hyperparameter training for RFC
        tuned_params = model_rfc.best_params_
        tuned_model_rf = RandomForestClassifier()
        tuned_model_rf.set_params(**tuned_params)
        hyp_start_time = time.time()
        tuned_model_rf.fit(X_train, y_train)
        logger.info('[RandomForestClassifier] Training Time with best hyper '
                    'parameters: %f secs', time.time()-hyp_start_time)
        best_fit_model = model_rfc
    elif trainmodel == "logistic_regression":
        model_lr = LogisticRegression()
        params = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(model_lr, param_grid=params, cv=3, verbose=10, n_jobs=-1)
        hyp_start_time = time.time()  # timing starts from this point for "hyp_start_time" variable
        lr_model_rs = grid_search.fit(X_train, y_train)
        logger.info('[LogisticRegression] Training Time with Hyper '
                    'parameter Tuning: %f secs', time.time()-hyp_start_time)
        logger.info('[LogisticRegression] Best parameters %s', lr_model_rs.best_params_)
        logger.info('[LogisticRegression] Best Accuracy score %s', lr_model_rs.best_score_)

        # Tuned hyper parameter training for LR
        tuned_params = lr_model_rs.best_params_
        tuned_model_lr = LogisticRegression()
        tuned_model_lr.set_params(**tuned_params)
        hyp_start_time = time.time()
        tuned_model_lr.fit(X_train, y_train)
        logger.info('[LogisticRegression] Training Time with best hyper '
                    'parameters: %f secs', time.time()-hyp_start_time)
        best_fit_model = lr_model_rs

    return best_fit_model.best_estimator_


if __name__ == "__main__":
    # Customer Churn Prediction using Random Forest Classifier and Logistic Regression
    # 1. Data preperation
    # 2. Hyper parameter Tuning
    # 3. Training with Dataset

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--datasize',
                        type=int,
                        required=False,
                        default=71893,
                        help='Dataset size, default is full dataset')
    parser.add_argument('-hy',
                        '--hyperparams',
                        type=int,
                        required=False,
                        default=0,
                        help='Enabling Hyperparameter tuning (0/1)')
    parser.add_argument('-tr',
                        '--training',
                        type=int,
                        required=False,
                        default=0,
                        help='Enabling training (0/1)')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=300,
                        help='Enabling inference (0/1)')
    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help="Use intel accelerated technologies where available (0/1)")
    parser.add_argument('-ts',
                        '--testsplit',
                        type=int,
                        required=False,
                        default=20,
                        help="Percentage of test split from the total dataset (default is 20)"
                             "Remaining percentage will be used as Training dataset split (default is 80)")
    parser.add_argument('-model',
                        '--modelname',
                        type=str,
                        required=False,
                        default="random_forest_classifier",
                        help="Default is 'random_forest_classifier'")
    parser.add_argument('-save',
                        '--savemodeldir',
                        type=str,
                        required=False,
                        default="models/customerchurn_rfc_joblib",
                        help="Please specify model path along with model name to save "
                             "Default is 'models/customerchurn_rfc_joblib'")
    FLAGS = parser.parse_args()
    data_max_size = FLAGS.datasize
    hyperparams_flag = FLAGS.hyperparams
    training_flag = FLAGS.training
    model = FLAGS.modelname
    save_model_dir = FLAGS.savemodeldir
    if FLAGS.testsplit >= 50:
        print("Test split is Morethan 50% of Dataset, Recommended to use the test split of less than 40%")
    test_split = (FLAGS.testsplit/100)

    if FLAGS.intel == 1:
        from sklearnex import patch_sklearn  # pylint: disable=import-error
        patch_sklearn()  # Patching should be called before first sklearn package import

    from sklearn.model_selection import GridSearchCV

    from sklearn.ensemble import RandomForestClassifier  # noqa:F811
    from sklearn.linear_model import LogisticRegression  # noqa:F811

    X_train, X_test, y_train, y_test = data_preparation(data_max_size, test_split)

    # Model Training
    if hyperparams_flag and training_flag:
        best_model = hyperparams_tuning(model)
        savemodel(best_model, save_model_dir)
    elif training_flag:
        if model == "random_forest_classifier":
            # RandomForestClassifier
            rfc_model = RandomForestClassifier()
            start_time = time.time()
            rfc_model.fit(X_train, y_train)
            logger.info('[RandomForestClassifier] Training time: %f secs', time.time()-start_time)
            savemodel(rfc_model, save_model_dir+"_default")
        elif model == "logistic_regression":
            # Logistic Regression
            lr_model = LogisticRegression()
            start_time = time.time()
            lr_model.fit(X_train, y_train)
            logger.info('[LogisticRegression] Training time: %f secs', time.time()-start_time)
            savemodel(lr_model, save_model_dir + "_default")
