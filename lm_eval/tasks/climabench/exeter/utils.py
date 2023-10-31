import datasets
import numpy as np
import sklearn.metrics

# cols_to_remove = ['id_qa', 'corpus', 'question_pt_origin', 'question_en_paraphase', 'question_pt_paraphase', 'answer_en_origin', 'answer_pt_origin', 'answer_en_validate', 'answer_pt_validate', 'eid_article_scopus', 'question_generic', 'answer_in_text', 'answer_difficulty', 'question_meaningful', 'answer_equivalent', 'question_type', 'abstract_translated_pt', 'pt_question_translated_to_en']

def f1(predictions, references):  # This is a passthrough function

    _prediction = predictions[0]
    _reference = references[0]
    print(f"_prediction: {_prediction}")
    print(f"_reference: {_reference}")
    string_label = ['0', '1']
    reference = string_label.index(_reference)
    prediction = (
        string_label.index(_prediction)
        if _prediction in string_label
        else not bool(reference)
    )

    return (prediction, reference)

def multi_f1(predictions, references):  # This is a passthrough function

    _prediction = predictions[0]
    _reference = references[0]
    print(f"_prediction: {_prediction}")
    print(f"_reference: {_reference}")
    string_label = ['0', '1', '2', '3', '4']
    reference = string_label.index(_reference)
    prediction = (
        string_label.index(_prediction)
        if _prediction in string_label
        else not bool(reference)
    )

    return (prediction, reference)

def agg_f1(items):

    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    return sklearn.metrics.f1_score(references, predictions, average='macro')


def relabel_labels(example):
    example["label"]= example["claim_labels"]-1
    return example

def process_labels(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.map(relabel_labels, batched=False)
    print(dataset)
    return dataset

# def convert_to_int(example):
#     example["label"]= 1 if example["at_labels"]==1.0 else 0
#     # print(example["label"])
#     return example
#     # return int(example)

# def prepate_AT_data(dataset: datasets.Dataset) -> datasets.Dataset:
#     # Only supports and refutes
#     print(dataset)
#     at_data = dataset.remove_columns(cols_to_remove)
#     # at_data = at_data.rename_column("question_en_origin", "question")
#     # at_data = at_data.rename_column("at_labels", "label")
#     # print(at_data)
#     data = at_data.filter(lambda example: example["at_labels"] is not None)
#     # print(data)
#     data = data.map(convert_to_int, batched=False)
#     # print(f"DATA after applying filter: {data['label'][:5]}")
#     return data


    # return dataset.filter(lambda example: example["claim_label"] in [0,1])