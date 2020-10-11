import copy
import itertools
import json
import random
import argparse
from hashlib import sha1


random.seed(102)


def shuffle_history(utterances, overwrite_hash=False):
    """Shuffles history if present."""
    augmented_utterances = []
    for utterance in utterances:
        history = utterance['history']
        permutations = list(itertools.permutations(history))
        random.shuffle(permutations)
        for history in permutations[:1]:
            new_utterance = copy.deepcopy(utterance)
            new_utterance['history'] = list(history)
            if overwrite_hash:
                del new_utterance['utterance_id']
                new_utterance['utterance_id'] = sha1(
                    str(new_utterance).encode('utf-8')).hexdigest()
            augmented_utterances.append(new_utterance)
    return augmented_utterances


def clean_dataset(dataset):
    fixed_dataset = copy.deepcopy(dataset)
    # correct spelling errors
    for utterance in fixed_dataset:
        for followup_qa in utterance['history'] + utterance['evidence']:
            if 'followup_question' in followup_qa:
                followup_qa['follow_up_question'] = followup_qa.pop(
                    'followup_question')
            if 'followup_answer' in followup_qa:
                followup_qa['follow_up_answer'] = followup_qa.pop(
                    'followup_answer')

    return fixed_dataset


def main(args):

    with open(args.train_in) as f:
        train_json = json.load(f)
    with open(args.dev_in) as f:
        dev_json = json.load(f)

    train_json = clean_dataset(train_json)
    train_json = shuffle_history(train_json, args.overwrite_hash)

    dev_json = clean_dataset(dev_json)
    dev_json = shuffle_history(dev_json, args.overwrite_hash)

    with open(args.train_out, 'w') as f:
        json.dump(train_json, f)
    with open(args.dev_out, 'w') as f:
        json.dump(dev_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default="data/sharc_train.json", help="path to train input file")
    parser.add_argument("--dev-in", default="data/sharc_dev.json", help="path to dev input file")
    parser.add_argument("--train-out", default="data/history_train.json", help="path to train output file")
    parser.add_argument("--dev-out", default="data/history_dev.json", help="path to dev output file")
    parser.add_argument("--overwrite-hash", action='store_true', help="whether to recalculate SHA1 hashes")
    args = parser.parse_args()

    main(args)
