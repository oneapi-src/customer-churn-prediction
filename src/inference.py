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
from utils import data_preparation, loadmodel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

NUM_ITERATIONS = 100


def run_realtime_inference(trained_model):
    """Running Real time inference for specified NUM_ITERATIONS"""
    total_time = 0
    for i in range(0, NUM_ITERATIONS):
        realtime_inference_start_time = time.time()
        trained_model.predict(X_test.head(i + 1))
        time_taken = time.time() - realtime_inference_start_time
        total_time += time_taken
    return total_time


if __name__ == "__main__":
    # Customer Churn Prediction using Random Forest Classifier and Logistic Regression
    # 1. Data preperation
    # 2. Inference

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--datasize',
                        type=int,
                        required=False,
                        default=71893,
                        help='Dataset size, default is full dataset')
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
    parser.add_argument('-model',
                        '--modelname',
                        type=str,
                        required=False,
                        default="random_forest_classifier",
                        help="Default is 'random_forest_classifier'")
    parser.add_argument('-save',
                        '--saved_model_dir',
                        type=str,
                        required=False,
                        default="models/customerchurn_rfc_joblib",
                        help="Please specify model path along with model name to save "
                             "Default is 'models/customerchurn_rfc_joblib'")
    parser.add_argument('-ts',
                        '--testsplit',
                        type=int,
                        required=False,
                        default=20,
                        help="Percentage of test split from the total dataset (default is 20)"
                             "Remaining percentage will be used as Training dataset split (default is 80)")
    FLAGS = parser.parse_args()
    data_max_size = FLAGS.datasize
    model_name = FLAGS.modelname
    modelfile = FLAGS.saved_model_dir
    batchsize = FLAGS.batchsize
    test_split = (FLAGS.testsplit/100)

    if FLAGS.intel == 1:
        from sklearnex import patch_sklearn  # pylint: disable=import-error
        patch_sklearn()  # Patching should be called before first sklearn package import

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X_train, X_test, y_train, y_test = data_preparation(data_max_size, test_split)

    # Initialization
    model = None
    # Load saved model
    if modelfile is not None:
        model = loadmodel(modelfile)

    # Using Test Dataset for Inference until batchsize exceeds test dataset limit
    if batchsize > X_test.shape[0]:
        X_test = X_train
        y_test = y_train
        logger.info('Using Training data for Inferencing as the batchsize is more than test data size')
    # Make predictions
    if model is not None:
        if model_name == "random_forest_classifier":
            prediction_test_warm = model.predict(X_test.head(batchsize))  # Warm up
            start_time = time.time()
            prediction_test = model.predict(X_test.head(batchsize))
            logger.info('[RandomForestClassifier] Time taken for Batch inference of size %s is: %f secs', batchsize,
                        time.time() - start_time)
            # Model evaluation
            logger.info('[RandomForestClassifier] Classification Report\n%s',
                        classification_report(y_test[:batchsize], prediction_test))
            logger.info('[RandomForestClassifier] Confusion Matrix\n%s',
                        confusion_matrix(y_test[:batchsize], prediction_test))
            logger.info('[RandomForestClassifier] Accuracy is: %f',
                        accuracy_score(y_test[:batchsize], prediction_test))

            totaltime = run_realtime_inference(model)
            logger.info('[RandomForestClassifier] Average Real Time inference: %f secs', (totaltime / NUM_ITERATIONS))

        elif model_name == "logistic_regression":
            lr_pred_warm = model.predict(X_test.head(batchsize))  # Warm up
            start_time = time.time()
            lr_pred = model.predict(X_test.head(batchsize))
            logger.info('[LogisticRegression] Time taken for Batch inference of size %s is: %f secs',
                        batchsize, time.time() - start_time)
            # Model evaluation
            logger.info('[LogisticRegression] Classification Report\n%s',
                        classification_report(y_test[:batchsize], lr_pred))
            logger.info('[LogisticRegression] Confusion Matrix\n%s',
                        confusion_matrix(y_test[:batchsize], lr_pred))
            logger.info('[LogisticRegression] Accuracy is: %f',
                        accuracy_score(y_test[:batchsize], lr_pred))

            totaltime = run_realtime_inference(model)
            logger.info('[LogisticRegression] Average Real Time inference: %f secs', (totaltime / NUM_ITERATIONS))
