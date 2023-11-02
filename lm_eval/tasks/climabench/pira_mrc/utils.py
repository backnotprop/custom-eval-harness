import datasets
import numpy as np
import sklearn.metrics
import string
import re
import collections

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


# Exact match (the normalized answer exactly match the gold answer)
def exact(predictions, references):
    return int(normalize_answer(references[0]) == normalize_answer(predictions[0]))


# The F-score of predicted tokens versus the gold answer
def f1(predictions, references):
    gold_toks = get_tokens(references[0])
    pred_toks = get_tokens(predictions[0])
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



# cols_to_remove = ['id_qa', 'corpus', 'question_pt_origin', 'question_en_paraphase', 'question_pt_paraphase', 'answer_en_origin', 'answer_pt_origin', 'answer_en_validate', 'answer_pt_validate', 'eid_article_scopus', 'question_generic', 'answer_in_text', 'answer_difficulty', 'question_meaningful', 'answer_equivalent', 'question_type', 'abstract_translated_pt', 'pt_question_translated_to_en']

# def normalize_answer(s):
#     """Lower text and remove punctuation and extra whitespace."""

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_punc(lower(s)))


# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


# def exact_match_score(prediction, ground_truth):
#     return (normalize_answer(prediction) == normalize_answer(ground_truth))


# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)


# def evaluate(references, predictions):
#     f1 = exact_match = total = 0

#     for ground_truths, prediction in zip(references, predictions):
#       total += 1
#       exact_match += metric_max_over_ground_truths(
#                     exact_match_score, prediction, ground_truths)
#       f1 += metric_max_over_ground_truths(
#           f1_score, prediction, ground_truths)
    
#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total
#     print(f"exact_match: {exact_match}")
#     print(f"f1: {f1}")
#     return f1

#     # return {'exact_match': exact_match, 'f1': f1}
# def agg_f1(f1_score):
#     print(f"f1_score: {f1_score}")

#     # predictions, references = zip(*items)
#     # references, predictions = np.asarray(references), np.asarray(predictions)

    # return sklearn.metrics.f1_score(references, predictions, average='macro')

# def f1(predictions, references):  # This is a passthrough function

#     _prediction = predictions[0]
#     _reference = references[0]
#     string_label = ['0', '1']
#     reference = string_label.index(_reference)
#     prediction = (
#         string_label.index(_prediction)
#         if _prediction in string_label
#         else not bool(reference)
#     )

#     return (prediction, reference)

# def agg_f1(items):

#     predictions, references = zip(*items)
#     references, predictions = np.asarray(references), np.asarray(predictions)

#     return sklearn.metrics.f1_score(references, predictions, average='macro')


# def convert_to_int(example):
#     # example["at_labels"] = int(example["at_labels"])
#     return int(example)

# def prepate_AT_data(dataset: datasets.Dataset) -> datasets.Dataset:
#     # Only supports and refutes
#     at_data = dataset.remove_columns(cols_to_remove)
#     # at_data = at_data.rename_column("question_en_origin", "question")
#     # at_data = at_data.rename_column("at_labels", "label")
#     # print(at_data)
#     data = at_data.filter(lambda example: example["at_labels"] is not None)
#     print(data)
#     # data = data.map(lambda example["at_labels"]: int(example["at_labels"]), batched=True)
#     # print(data)
#     return data


#     # return dataset.filter(lambda example: example["claim_label"] in [0,1])