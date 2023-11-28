import datasets
import numpy as np
import logging
from sklearn.metrics import f1_score

def f1(predictions, references):  # This is a passthrough function
    return (predictions[0], references[0])
    # _prediction = predictions[0]
    # _reference = references[0]
    # print(f"_prediction: {_prediction}")
    # print(f"_reference: {_reference}")
    # string_label = ['0', '1', '2', '3', '4']
    # reference = string_label.index(_reference)
    # prediction = (
    #     string_label.index(_prediction)
    #     # if _prediction in string_label
    #     # else not bool(reference)
    # )

    # return (prediction, reference)

def agg_f1(items):

    predictions, references = zip(*items)
    logging.info(f"predictions: {predictions}")
    logging.info(f"references: {references}")
    references, predictions = np.asarray(references), np.asarray(predictions)

    return f1_score(references, predictions, average='macro')
