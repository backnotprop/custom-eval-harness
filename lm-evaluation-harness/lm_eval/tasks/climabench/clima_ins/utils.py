import datasets


def filter_multi(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.filter(lambda example: example["label"] in [0,1])