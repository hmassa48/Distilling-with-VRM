import numpy as np
import sklearn.metrics as metrics


def bin_accuracy(conf_lower_thresh, conf_upper_thresh, conf, pred, true):
    accuracy = avg_confidence = len_bin = 0
    filtered_tuples = [
        x
        for x in zip(pred, true, conf)
        if x[2] > conf_lower_thresh and x[2] <= conf_upper_thresh
    ]
    if len(filtered_tuples) > 0:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        len_bin = len(filtered_tuples)
        avg_confidence = sum([x[2] for x in filtered_tuples]) / len_bin
        accuracy = float(correct) / len_bin

    return accuracy, avg_confidence, len_bin


def CalibrationError(conf, pred, true, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    num_points = len(conf)
    ece = mce = 0
    accuracies = [0.0]
    thresholds = [0.0]

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = bin_accuracy(
            conf_thresh - bin_size, conf_thresh, conf, pred, true
        )
        accuracies.append(acc)
        thresholds.append(conf_thresh)
        ece += (np.abs(acc - avg_conf)) * (len_bin / num_points)
        mce = max(mce, np.abs(acc - avg_conf))

    elem = {
        "MCE" : mce,
        "ECE" : ece,
        "Thresholds": thresholds,
        "Accuracies": accuracies
    }

    return elem