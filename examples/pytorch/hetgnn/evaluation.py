# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Functions to evaluate HetGnn for different tasks."""
import numpy
import csv
from sklearn import linear_model, metrics


def load_data(data_file_name, n_features, n_samples):
    """Load data from a CSV file."""
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        for i, d in enumerate(data_file):
            data[i] = numpy.asarray(d[:], dtype=numpy.float32)
        f.close

        return data


def evaluate_node_classification(train_num, test_num, config):
    """Evaluate HetGnn model on a node classification task."""
    train_data_f = str(config["data_dir"] + ("/train_class_feature.txt"))
    train_data = load_data(train_data_f, config["dim"] + 2, train_num)
    train_features = train_data.astype(numpy.float32)[:, 2:-1]
    train_target = train_data.astype(numpy.float32)[:, 1]

    learner = linear_model.LogisticRegression()
    learner.fit(train_features, train_target)
    train_features = None
    train_target = None

    test_data_f = str(config["data_dir"] + ("/test_class_feature.txt"))
    test_data = load_data(test_data_f, config["dim"] + 2, test_num)
    test_id = test_data.astype(numpy.int32)[:, 0]
    test_features = test_data.astype(numpy.float32)[:, 2:-1]
    test_target = test_data.astype(numpy.float32)[:, 1]
    test_predict = learner.predict(test_features)
    test_features = None

    output_f = open(str(config["data_dir"] + ("/NC_prediction.txt")), "w")
    for i in range(len(test_predict)):
        output_f.write("%d,%lf\n" % (test_id[i], test_predict[i]))
    output_f.close()

    return (
        metrics.f1_score(test_target, test_predict, average="macro"),
        metrics.f1_score(test_target, test_predict, average="micro"),
    )


def evaluate_link_prediction(config, train_num, test_num):
    """Evaluate HetGnn model for a link prediction task."""
    # prepare training data and train.
    train_data_f = str(config["data_dir"] + ("/train_feature.txt"))
    train_data = load_data(train_data_f, config["dim"] + 3, train_num)
    train_features = train_data.astype(numpy.float32)[:, 3:-1]
    train_target = train_data.astype(numpy.float32)[:, 2]

    learner = linear_model.LogisticRegression(random_state=0)
    learner.fit(train_features, train_target)
    train_features = None
    train_target = None

    # prepare test data
    test_data_f = str(config["data_dir"] + ("/test_feature.txt"))
    test_data = load_data(test_data_f, config["dim"] + 3, test_num)
    test_id = test_data.astype(numpy.int32)[:, 0:2]
    test_features = test_data.astype(numpy.float32)[:, 3:-1]
    test_target = test_data.astype(numpy.float32)[:, 2]
    test_predict = learner.predict(test_features)
    test_features = None

    output_f = open(str(config["data_dir"] + ("/link_prediction.txt")), "w")
    for i in range(len(test_predict)):
        output_f.write(
            "%d, %d, %lf\n" % (test_id[i][0], test_id[i][1], test_predict[i])
        )
    output_f.close()

    auc_score = metrics.roc_auc_score(test_target, test_predict)

    total_count = 0
    correct_count = 0
    true_p_count = 0
    false_p_count = 0
    false_n_count = 0

    for i in range(len(test_predict)):
        total_count += 1
        if int(test_predict[i]) == int(test_target[i]):
            correct_count += 1
        if int(test_predict[i]) == 1 and int(test_target[i]) == 1:
            true_p_count += 1
        if int(test_predict[i]) == 1 and int(test_target[i]) == 0:
            false_p_count += 1
        if int(test_predict[i]) == 0 and int(test_target[i]) == 1:
            false_n_count += 1

    precision = float(true_p_count) / (true_p_count + false_p_count)
    recall = float(true_p_count) / (true_p_count + false_n_count)
    F1 = float(2 * precision * recall) / (precision + recall)

    return auc_score, F1
