from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn import metrics, preprocessing
from sklearn.utils.fixes import loguniform
import scipy.stats as stats
from sklearn.utils import resample
import pandas as pd
import numpy as np
#import mkl
import os

#mkl.set_num_threads(1)  # control the number of thread used for NN model
N_WORKERS = 1  # control the number of workers used for RF model

INPUT_FOLDER = r'./'
GOOGLE_INPUT_FILE = r'../../../Documents/phd_related/data_sets_concept_drift/AIOps_failure_prediction/google_job_failure.csv'
BACKBLAZE_INPUT_FILE = r'disk_failure_v2.csv'
ALIBABA_INPUT_FILE = r'../../../Documents/phd_related/data_sets_concept_drift/AIOps_failure_prediction/alibaba_job_data.csv'
WINDOW_HYPER_PARAMETER_FILE = r'parameter_list_window.csv'
PERIOD_HYPER_PARAMETER_FILE = r'parameter_list_period.csv'


def obtain_tuned_model(model_name, dataset, period, mode):
    if dataset == 'g':
        dataset = 'Google'
    elif dataset == 'b':
        dataset = 'Backblaze'
    elif dataset == 'a':
        dataset = 'Alibaba'

    if mode == 'w': # half of total periods window, for concept drift detection
        df = pd.read_csv(WINDOW_HYPER_PARAMETER_FILE)
        # i + 1 in features = np.vstack(feature_list[i-window_size: i])
    elif mode == 'p': # single time period, for ensemble
        df = pd.read_csv(PERIOD_HYPER_PARAMETER_FILE)
    else:
        return None

    para_str = df[np.logical_and(np.logical_and(df['Dataset']==dataset, df['Period']==period), df['Model']==model_name)].iloc[0]['Hyper']
    para_dic = eval(para_str)

    if model_name == 'lr':
        model = LogisticRegression(**para_dic)
    elif model_name == 'cart':
        model = DecisionTreeClassifier(**para_dic)
    elif model_name == 'gbdt':
        model = XGBClassifier(n_jobs=N_WORKERS, **para_dic)
    elif model_name == 'nn':
        model = MLPClassifier(**para_dic)
    elif model_name == 'rf':
        model = RandomForestClassifier(n_jobs=N_WORKERS, **para_dic)
    return model


def obtain_param_dist(model_name):
    param_dist = None

    if model_name == 'lr':
        param_dist = [
            {'solver': ['newton-cg', 'lbfgs', 'sag'], 'C': loguniform(1e-2, 1e2), 'penalty': ['l2', 'none']},
            {'solver': ['saga'], 'C': loguniform(1e-2, 1e2), 'penalty': ['l1', 'l2', 'none', 'elasticnet']},
            {'solver': ['liblinear'], 'C': loguniform(1e-2, 1e2), 'penalty': ['l1', 'l2']}
        ]
    elif model_name == 'cart':
        param_dist = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'class_weight':['balanced', None]
        }
    elif model_name == 'gbdt':
        param_dist = {
            'n_estimators': stats.randint(1e1, 1e2),
              'learning_rate': stats.uniform(1e-2, 1),
              'max_depth': stats.randint(2, 10),
              'subsample': loguniform(5e-1, 1),
              'booster': ['gbtree', 'gblinear', 'dart']
             }
    elif model_name == 'nn':
        param_dist = {
            'hidden_layer_sizes': [(8,), (16,), (32,), 
                                   (8, 8), (16, 16),],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': loguniform(1e-4, 1e-2),
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'max_iter': stats.randint(1e1, 2e2)
        }
    elif model_name == 'rf':
        param_dist = {
            'n_estimators': stats.randint(1e1, 1e2),
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=6)] + [None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 4, 8],
            'class_weight':['balanced', None],
            'bootstrap': [True, False]
        }
    return param_dist


def obtain_raw_model(model_name):
    if model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'cart':
        model = DecisionTreeClassifier()
    elif model_name == 'gbdt':
        model = XGBClassifier(n_jobs=1)
    elif model_name == 'nn':
        model = MLPClassifier()
    elif model_name == 'rf':
        model = RandomForestClassifier()
    return model


def obtain_data(dataset, interval='m'):
    if dataset == 'g':
        return get_google_data()
    elif dataset == 'b':
        return get_disk_data(interval)
    elif dataset == 'a':
        return get_alibaba_data()


def obtain_feature_names(dataset):
    if dataset == 'g':
        return ['User ID', 'Job Name', 'Scheduling Class',
               'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
               'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    elif dataset == 'b':
        return ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
        'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']


def get_google_data():
    '''
    Read the Google dataset from csv file
    The input path and filename are specified in macros
    Return features and labels after proper preprocessing

    Returns:
        features (np.array): feature vector, the first column is the timestamp
        labels (np.array): True or False, binary classification
    '''
    path = os.path.join(INPUT_FOLDER, GOOGLE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)

    columns = ['Start Time', 'User ID', 'Job Name', 'Scheduling Class',
               'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
               'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    print('Load complete')

    features = df[columns].to_numpy()
    labels = (df['Status']==3).to_numpy()

    print('Preprocessing features')
    le = preprocessing.LabelEncoder()
    features[:, 1] = le.fit_transform(features[:, 1])

    le = preprocessing.LabelEncoder()
    features[:, 2] = le.fit_transform(features[:, 2])
    print('Preprocessing complete\n')

    return features, labels


def get_disk_data(interval='d', production=None):
    '''
    Read the Backblaze disk dataset from csv file
    The input path and filename are specified in macros
    Return features and labels after proper preprocessing
    
    Args:
        interval (chr): the interval of timestamp, by default day of year (d)
        Possible selections are day in a year (d) and month in a year (m)

    Returns:
        features (np.array): feature vector, the first column is the timestamp
        labels (np.array): True or False, binary classification
    '''
    path = os.path.join(INPUT_FOLDER, BACKBLAZE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)
    print('Load complete')
    
    print('Preprocessing features')
    df = df[['date',
        'smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
        'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff',
        'label']]
    # change the date into days of a year as all data are in 2015
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    if interval == 'd':
        df['date'] = pd.Series(pd.DatetimeIndex(df['date']).dayofyear)
    elif interval == 'm':
        df['date'] = pd.Series((pd.DatetimeIndex(df['date']).year - 2015) * 12 + pd.DatetimeIndex(df['date']).month)
    else: 
        print('Invalid time interval argument for reading disk failure data. Possible options are (d, m).')
        exit(-1)
    
    features = df[df.columns[:-1]].to_numpy()
    labels = df[df.columns[-1]].to_numpy()

    return features, labels


def get_alibaba_data():
    path = os.path.join(INPUT_FOLDER, ALIBABA_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)

    columns = ['start_time', 
        'user', 'task_name', 'inst_num', 'plan_cpu', 'plan_mem', 'plan_gpu', 
        'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']
    print('Load complete')

    features = df[columns].to_numpy()
    labels = (df['status']=='Failed').to_numpy()

    print('Preprocessing features')
    le = preprocessing.LabelEncoder()
    features[:, 1] = le.fit_transform(features[:, 1])
    print('Preprocessing complete\n')

    return features, labels


def obtain_metrics(labels, probas):
    '''
    Calculate performance on various metrics

    Args: 
        labels (np.array): labels of samples, should be True/False
        probas (np.array): predicted probabilities of samples, should be in [0, 1]
            and should be generated with predict_proba()[:, 1]
    Returns:
        (list): [ Precision, Recall, Accuracy, F-Measure, AUC, MCC, Brier Score ]
    '''
    ret = []
    preds = probas > 0.5
    auc = metrics.roc_auc_score(labels, probas)
    ret.append(metrics.precision_score(labels, preds))
    ret.append(metrics.recall_score(labels, preds))
    ret.append(metrics.accuracy_score(labels, preds))
    ret.append(metrics.f1_score(labels, preds))
    ret.append(np.max(auc, 1.0 - auc))
    ret.append(metrics.matthews_corrcoef(labels, preds))
    ret.append(metrics.brier_score_loss(labels, probas))

    return ret
    

def downsampling(training_features, training_labels, ratio=10):
    #return training_features, training_labels

    idx_true = np.where(training_labels == True)[0]
    idx_false = np.where(training_labels == False)[0]
    
    #print('Before dowmsampling:', len(idx_true), len(idx_false))
    if len(idx_true)*ratio >= len(idx_false):
        return training_features, training_labels

    idx_false_resampled = resample(idx_false, n_samples=len(idx_true)*ratio, replace=False)
    idx_resampled = np.concatenate([idx_false_resampled, idx_true])
    idx_resampled.sort()
    resampled_features = training_features[idx_resampled]
    resampled_labels = training_labels[idx_resampled]
    #print('After dowmsampling:', len(idx_true), len(idx_false_resampled))
    return resampled_features, resampled_labels


def time_based_splitting(features, labels, ratio):
    '''
    Split the data according to their timestamp
    Note that it assumes the first column is the timestamp
    '''
    count = int(np.round(len(features) * ratio))
    sort = features[:, 0].astype(np.int64).argsort()
    first_indexes = sort[:count]
    second_indexes = sort[count:]
    training_features = features[first_indexes]
    testing_features = features[second_indexes]
    training_labels = labels[first_indexes]
    testing_labels = labels[second_indexes]

    return training_features, testing_features, training_labels, testing_labels
    
    
def obtain_intervals(dataset):
    '''
    Generate interval terminals, so that samples in each interval have:
        interval_i = (timestamp >= terminal_i) and (timestamp < terminal_{i+1})

    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
    '''
    if dataset == 'g':
        # time unit in Google: millisecond, tracing time: 28 days
        start_time = 604046279
        unit_period = 24 * 60 * 60 * 1000 * 1000  # unit period: one day
        end_time = start_time + 28*unit_period
    elif dataset == 'b':
        # time unit in Backblaze: month, tracing time: 3 years (36 months)
        start_time = 1
        unit_period = 1  # unit period: one month
        end_time = start_time + 36*unit_period
    elif dataset == 'a':
        # time unit in Alibaba: second, tracing time: 
        start_time = 494319
        unit_period = 7 * 24 * 60 * 60  # unit period: one week
        start_time += 3 * 24 * 60 * 60  # the first week contains only 1642 samples
        end_time = start_time + 8*unit_period

    # add one unit for the open-end of range function
    terminals = [i for i in range(start_time, end_time+unit_period, unit_period)]

    return terminals
    

# obtain data in natural time periods, the timestamp column is stripped
def obtain_period_data(dataset):
    features, labels = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)
    feature_list = []
    label_list = []

    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        feature_list.append(features[idx][:, 1:])
        label_list.append(labels[idx])
    return feature_list, label_list


def obtain_chunks(features, labels, N):
    '''
    Split data into N consecutive chunks
    If the size of the last chunk is smaller, it will be merged into the second last chunk
    Return a list of chunks for features and labels
    '''
    feature_list = []
    label_list = []
    n_samples = features.shape[0] // N
    for i in range(N):
        if i != N - 1:
            feature_list.append(features[i*n_samples: (i + 1)*n_samples])
            label_list.append(labels[i*n_samples: (i + 1)*n_samples])
        else:
            feature_list.append(features[i*n_samples:])
            label_list.append(labels[i*n_samples:])
            break

    print([len(label) for label in label_list])
    return feature_list, label_list
