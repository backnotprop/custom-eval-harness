import datasets


def select_first_1100(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.shuffle(42).select(range(1100))
