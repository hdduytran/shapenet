import os
import json
import math
import torch
import numpy as np
import pandas as pd
import argparse
from TimeSeries import TimeSeries
import timeit
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from pyts.datasets import fetch_uea_dataset
import wrappers_original


def load_UEA_dataset(dataset_name, train_ratio=0.9):
    """
    Loads the UEA dataset given in input in np arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    
    uv_dir = "./data/" 
    data = fetch_uea_dataset(dataset_name, data_home=uv_dir)
    Train_dataset = data['data_train']
    Test_dataset = data['data_test']
    Train_dataset = Train_dataset.astype(np.double)
    Test_dataset = Test_dataset.astype(np.double)
    

    X_train = Train_dataset
    y_train = data['target_train']

    X_test = Test_dataset
    y_test = data['target_test']
    print('y_train: ',y_train.shape)
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    if train_ratio < 1:
        X_train_ori, y_train_ori = X_train, y_train
        sss = StratifiedShuffleSplit(n_splits=10, test_size=1 - train_ratio)
        sss.get_n_splits(X_train, y_train)

        for train_index, test_index in sss.split(X_train_ori, y_train_ori):
            X_train = X_train_ori[train_index,:]
            y_train = y_train_ori[train_index]
    
        print(f'Ratio {train_ratio} - train shape: {np.shape(y_train)}')
    print('dataset load succeed !!!')
    return X_train, y_train, X_test, y_test


def fit_parameters(file, ratio, ind, train, train_labels, test, test_labels, cuda, gpu, save_path, cluster_num,
                        save_memory=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = wrappers_original.CausalCNNEncoderClassifier()

    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        ratio,ind, train, train_labels, test, test_labels, save_path, cluster_num, save_memory=save_memory, verbose=True
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=False,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')
    parser.add_argument('--ratio', type=float, default=1,
                        help='percent of training samples used for few-shot learning')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False
    ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    ind_list = [0, 1, 2]
    for ratio in ratio_list:
        for ind in ind_list:
            try:
                train, train_labels, test, test_labels = load_UEA_dataset(
                    args.dataset, ratio
                )
                cluster_num = 100
                if not args.load and not args.fit_classifier:
                    print('start new network training')
                    save_path = args.save_path + '/' + args.dataset + '.txt'
                    classifier = fit_parameters(
                    args.hyper, ratio, ind, train, train_labels, test, test_labels, args.cuda, args.gpu, save_path, cluster_num
                    )
                else:
                    classifier = wrappers_original.CausalCNNEncoderClassifier()
                    hf = open(
                        os.path.join(
                            args.save_path, args.dataset + '_parameters.json'
                        ), 'r'
                    )
                    hp_dict = json.load(hf)
                    hf.close()
                    hp_dict['cuda'] = args.cuda
                    hp_dict['gpu'] = args.gpu
                    classifier.set_params(**hp_dict)
                    classifier.load(os.path.join(args.save_path, args.dataset))

                if not args.load:
                    if args.fit_classifier:
                        classifier.fit_classifier(classifier.encode(train), train_labels)
                    classifier.save(
                        os.path.join(args.save_path, args.dataset), ratio
                    )
                    with open(
                        os.path.join(
                            args.save_path, args.dataset + '_parameters.json'
                        ), 'w'
                    ) as fp:
                        json.dump(classifier.get_params(), fp)
            except Exception as e:
                print(e)
                print(f'ratio {ratio} error')
                continue

    end = timeit.default_timer()
    print("All time: ", (end- start)/60)
