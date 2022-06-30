import os
import re
import faiss
import random
from transformers import AdapterConfig
from transformers.adapters.composition import Stack


MAX_LENGTH = 512

model_name = "bert-base-multilingual-cased"


def random_baseline(source, target, k=1):
    """
    Provided source and target sentences it creates a random baseline.

    :param source:
    :param target:
    :param k: the number of choices returned
    :return:
    """
    random_preds = []

    target_ids = range(len(target))

    for sentence in source:
        choice = random.choices(target_ids, k=k)
        random_preds.append(choice)

    return random_preds


def get_scores_predictions(x, y, dim=768, k=1):
    """
    Calculates cosine similarities or Euclidean distances of two groups of sentences.

    :param x: source sentences
    :param y: target sentences
    :param dim: dimensionality of the model (768 by default)
    :param k: how many results to return
    :return: a tuple of scores and predictions
    """
    idx = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)
    idx.add(x)
    scores, predictions = idx.search(y, k)
    return scores, predictions


# function from XTREME
def accuracy(labels, predictions, language=None):
    correct = sum([int(p == l) for p, l in zip(predictions, labels)])
    accuracy_score = float(correct) / len(predictions)
    return {'accuracy': round(accuracy_score * 100, 2)}


def k_3_accuracy(labels, predictions):
    correct = sum([int(l in p) for p, l in zip(predictions, labels)])
    accuracy_score = float(correct) / len(predictions)
    return {'k_3_accuracy': round(accuracy_score * 100, 2)}


def reciprocal_rank(gold_label, ranked_answers):
    if gold_label not in ranked_answers:
        rec_rank = 0
    else:
        rank = ranked_answers.index(gold_label) + 1
        rec_rank = 1 / rank
    return rec_rank


def create_test_set(source_file,
                    target_file,
                    output_dir,
                    move_source=True,
                    random_seed=15,
                    threshold=None):
    """
    Takes a source and target file from the Tatoeba challenge test sets,
    shuffles the target file, and outputs the shuffled set and the
    correct indices in separate files.

    :param source_file: test.src file
    :param target_file: test.trg file
    :param output_dir: language pair name
    :param move_source: if True, the test.src file will be moved to the same directory as shuffled.trg and the indices
    :param random_seed: 15 by default for reproducibility
    :param threshold: if set, an assertion error is thrown if it is not met by the test set
    :return: None
    """
    random.seed(random_seed)

    out_dir = 'data/output/'

    with open(source_file, 'r') as infile:
        source_set = infile.read().split('\n')[:-1]

    with open(target_file, 'r') as infile:
        target_set = infile.read().split('\n')[:-1]

    if threshold:
        assert len(source_set) >= threshold, f"Test set is lower than threshold ({threshold})!"

    target_set = [(sentence, idx) for idx, sentence in enumerate(target_set)]
    random.shuffle(target_set)

    idx = [idx for sentence, idx in target_set]
    target_set = [sentence for sentence, idx in target_set]

    output_dir = os.path.join(out_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trg_filepath = os.path.join(output_dir, "shuffled.trg")
    trg_labels = os.path.join(output_dir, "indices.txt")

    with open(trg_filepath, 'w') as outfile:
        for sentence in target_set:
            outfile.write(sentence)
            outfile.write('\n')

    with open(trg_labels, 'w') as outfile:
        for label in idx:
            outfile.write(str(label))
            outfile.write('\n')

    if move_source is True:

        src_filepath = os.path.join(output_dir, "source.src")

        with open(src_filepath, 'w') as outfile:
            for sentence in source_set:
                outfile.write(sentence)
                outfile.write('\n')


def load_lang_adapters(lang_adapters, model):
    lang_adapter_config = AdapterConfig.load("pfeiffer")
    for adapter in lang_adapters:
        model.load_adapter(f"{adapter}/wiki@ukp",
                           config=lang_adapter_config,
                           with_head=False)
        model.delete_head(adapter)

    if len(lang_adapters) > 1:
        model.set_active_adapters(Stack(*lang_adapters))
    else:
        model.set_active_adapters(*lang_adapters)

    print("Adapter setup   {}   added and activated".format(model.active_adapters))

    # print("Adapters {} added and activated.".format(*lang_adapters))


def load_config_list(config_line, multiconfig=False):
    if multiconfig is False:
        return config_line.split()
    else:
        return [split_line.split(',') for split_line in config_line.split()]
