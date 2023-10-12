import datasets


def filter_binary(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.filter(lambda example: example["label"] in [0,1])