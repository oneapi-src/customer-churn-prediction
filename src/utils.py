"""This code base is adopted from the below notebook
https://www.kaggle.com/code/edwingeevarughese/internet-service-churn-analysis
'''"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0209,C0301,W0108,C0103,E1101,E1137,E1136
import sys
import time
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATASET_FILE = 'data/internet_service_churn.csv'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def loadmodel(modelfile):
    """Loading the saved joblib model"""
    try:
        load_model = joblib.load(modelfile)
    except Exception as excep:
        raise IOError("Error loading model data from disk: {}".format(str(excep))) from excep
    return load_model


def savemodel(save_model, modelfile):
    """Saving the joblib model"""
    try:
        joblib.dump(save_model, modelfile)
    except Exception as exp:
        raise IOError("Error saving model data to disk: {}".format(str(exp))) from exp


def data_preparation(data_max_size, test_split):
    """Data Preparation """
    # Data preparation Starts here
    start_time = time.time()
    # loading data
    try:
        df = pd.read_csv(DATASET_FILE)
    except IOError:  # noqa:F841
        sys.exit('Dataset file not found')

    # Creating is_contract column
    df['is_contract'] = df['reamining_contract'].apply(lambda ele: 0 if pd.isna(ele) else 1)
    # Imputing null values with 0
    df['reamining_contract'].replace(np.nan, 0, inplace=True)
    # Rearranging columns
    column_names = ['id', 'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age', 'bill_avg',
                    'reamining_contract',
                    'is_contract', 'service_failure_count', 'download_avg', 'upload_avg', 'download_over_limit',
                    'churn']

    df = df.reindex(columns=column_names)

    df['download_avg'].replace('', np.nan, inplace=True)
    df['upload_avg'].replace('', np.nan, inplace=True)
    df.dropna(subset=['download_avg', 'upload_avg'], inplace=True)

    # Restructuring as per the correlation across the features
    df.corr()['churn'].sort_values(ascending=False)

    # Splitting up training and Label features for training
    x = df.drop(columns=['churn'])
    y = df['churn'].values

    if data_max_size != x.shape[0]:
        x = x.head(data_max_size)
        y = y[:data_max_size]
    logger.info('[Data] DataPreparation Time Taken in seconds --> %f secs', time.time() - start_time)
    logger.info('[Data] Total Data samples ---> %s', x.shape[0])

    # Preparing dataset for Training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=40, stratify=y)
    # Since the numerical features are distributed over different value ranges,
    # standard scalar is used to scale them down to the same range.
    num_cols = ['subscription_age', 'reamining_contract', 'download_avg', 'upload_avg',
                'download_over_limit', 'bill_avg', 'service_failure_count']

    scaler = StandardScaler()

    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])
    return x_train, x_test, y_train, y_test
