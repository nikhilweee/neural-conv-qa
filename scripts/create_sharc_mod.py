# Sharc shuffled Dataset
#
# There are a few major patterns to break.
# * When scenario is empty, the answer is irrelevant.
#   To break this, we randomly insert a new scenario
# * In case of conjunctive or disjunctive clauses,
#   the answer to the instance is the answer to the
#   last question in history. To break this, we randomly
#   shuffle the dialog history wherever possible.
#
# Some points to keep in mind:
# * Whenever shuffled, do multi bleu evaluation
# * Use a fixed seed for reproducability. Say 37.

import json
import copy
import random
import argparse
import numpy as np
from collections import defaultdict

random.seed(37)


counts = {
    'history_shuffle_successful': 0,
    'snippet_shuffle_successful': 0,
    'related_scenario_successful': 0,
    'history_shuffle': 0,
    'snippet_shuffle': 0,
    'related_scenario': 0,
    'both_shuffle': 0
}


def print_and_reset_counts():

    print(counts)

    for key in counts:
        counts[key] = 0


def derange(iterable):

    indices = list(range(len(iterable)))

    if len(indices) > 1:
        while any([idx == value for idx, value in enumerate(indices)]):
            random.shuffle(indices)

    deranged = [iterable[idx] for idx in indices]

    return deranged


def shuffle_snippet(snippet):

    sentences = snippet.split("\n")
    star_indices = []

    for idx, sentence in enumerate(sentences):
        if sentence.startswith("*"):
            star_indices.append(idx)

    if len(star_indices) > 1:
        start, end = star_indices[0], star_indices[-1] + 1
        # assert that star_indices are continuous
        assert end - start == len(star_indices)
        star_sentences = sentences[start:end]
        shuffled_stars = derange(star_sentences)
        sentences[start:end] = shuffled_stars
        counts['snippet_shuffle_successful'] += 1

    counts['snippet_shuffle'] += 1
    return "\n".join(sentences)


def shuffle_history(history):

    counts['history_shuffle'] += 1
    if len(history) > 1:
        counts['history_shuffle_successful'] += 1
    return derange(history)


def related_scenario(instance, tree_ids):

    tree_id = instance["tree_id"]
    counts['related_scenario'] += 1

    for instance in derange(tree_ids[tree_id]):
        if instance["scenario"]:
            counts['related_scenario_successful'] += 1
            return instance["scenario"]

    return instance['scenario']


def modify_instance(original_instance, tree_ids):

    modified_instance = copy.deepcopy(original_instance)

    history = modified_instance["history"]
    scenario = modified_instance["scenario"]
    answer = modified_instance["answer"].lower()
    snippet = modified_instance["snippet"]

    if not history:
        if answer in ["yes", "no"]:
            if random.random() > 0.5:
                modified_instance["snippet"] = shuffle_snippet(snippet)
        if answer in ["irrelevant"]:
            if not scenario:
                if random.random() > 0.5:
                    modified_instance["scenario"] = related_scenario(
                        modified_instance, tree_ids)
            else:
                if random.random() > 0.5:
                    modified_instance["snippet"] = shuffle_snippet(snippet)
    else:
        if answer in ["yes", "no", "irrelevant"]:

            if random.random() > 0.5:
                modified_instance["snippet"] = shuffle_snippet(snippet)
                snippet_modified = True
            else:
                snippet_modified = False

            if random.random() > 0.5:
                modified_instance["history"] = shuffle_history(history)
                history_modified = True
            else:
                history_modified = False

            if snippet_modified and history_modified:
                counts['both_shuffle'] += 1
                counts['history_shuffle'] -= 1
                counts['snippet_shuffle'] -= 1

    return modified_instance


def create_shuffled_dataset(dataset):

    tree_ids = defaultdict(list)
    dataset_shuffled = []

    for instance in dataset:
        tree_id = instance["tree_id"]
        tree_ids[tree_id].append(instance)

    for instance in dataset:
        new_instance = modify_instance(instance, tree_ids)
        dataset_shuffled.append(new_instance)

    return dataset_shuffled


def main(args):

    with open(args.train_in, "r") as f:
        train = json.load(f)
    with open(args.dev_in, "r") as f:
        dev = json.load(f)

    train_shuffled = create_shuffled_dataset(train)
    print_and_reset_counts()
    dev_shuffled = create_shuffled_dataset(dev)
    print_and_reset_counts()

    with open(args.train_out, 'w') as f:
        json.dump(train_shuffled, f)
    with open(args.dev_out, 'w') as f:
        json.dump(dev_shuffled, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default="data/sharc_train.json", help="path to train input file")
    parser.add_argument("--dev-in", default="data/sharc_dev.json", help="path to dev input file")
    parser.add_argument("--train-out", default="data/mod_train.json", help="path to train output file")
    parser.add_argument("--dev-out", default="data/mod_dev.json", help="path to dev output file")
    args = parser.parse_args()

    main(args)
