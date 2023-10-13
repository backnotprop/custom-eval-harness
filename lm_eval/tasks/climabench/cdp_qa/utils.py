import datasets

def select_first_1100(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.shuffle(42).select(range(1100))

def filter_multi(dataset: datasets.Dataset) -> datasets.Dataset:
    # Only supports and refutes
    return dataset.filter(lambda example: example["label"] in [0,1])

def filter_main(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = select_first_1100(dataset)
    return filter_multi(dataset)
