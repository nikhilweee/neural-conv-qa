import os
import sys
import json
import collections
import argparse
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import spacy


nlp = spacy.load("en_core_web_md")


class ClassificationEvaluator:
    def __init__(self, labels=None):
        self.labels = labels

    def evaluate(self, y_true, y_pred):
        if not self.labels:
            self.labels = list(set(y_true))

        # micro_accuracy = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
        micro_accuracy = accuracy_score(y_true, y_pred)
        results = {}
        results["micro"] = float(
            "{0:.4f}".format(micro_accuracy)
        )  # int(100 * micro_accuracy) / 100

        conf_mat = confusion_matrix(y_true, y_pred, labels=self.labels)
        conf_mat_norm = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean(
            [conf_mat_norm[i][i] for i in range(conf_mat_norm.shape[0])]
        )
        results["macro"] = float(
            "{0:.4f}".format(macro_accuracy)
        )  # int(100 * macro_accuracy) / 100
        return results


class MoreEvaluator:
    def __init__(self, max_bleu_order=4, bleu_smoothing=True):
        self.max_bleu_order = max_bleu_order
        self.bleu_smoothing = bleu_smoothing

    def evaluate(self, y_true, y_pred, prefix="bleu"):
        results = {}
        bleu_scores = [
            compute_bleu(
                [[y.split() for y in yt] for yt in y_true],
                [y.split() for y in y_pred],
                max_order=bleu_order,
                smooth=self.bleu_smoothing,
            )[0]
            for bleu_order in range(1, self.max_bleu_order + 1)
        ]

        for bleu_order, bleu_score in enumerate(bleu_scores):
            results[prefix + str(bleu_order + 1)] = float("{0:.4f}".format(bleu_score))
        return results


class CombinedEvaluator:
    def __init__(
        self,
        labels=["yes", "no", "more", "irrelevant"],
        accuracy_targets=["yes", "no", "irrelevant"],
    ):
        self.labels = labels
        self.accuracy_targets = accuracy_targets
        self.classification_evaluator = ClassificationEvaluator(labels=labels)
        self.more_evaluator = MoreEvaluator()

    def replace_follow_up_with_more(self, y_list):
        more_list = []
        for yl in y_list:
            if isinstance(yl, list):
                yl = [
                    y.lower() if y.lower() in self.accuracy_targets else "more"
                    for y in yl
                ]
                yl_uniques = list(set(yl))
                assert len(yl_uniques) == 1, "Conflicting labels are not allowed."
                yl = yl_uniques[0]
            else:
                yl = yl.lower() if yl in self.accuracy_targets else "more"
            more_list.append(yl)
        return more_list

    def extract_follow_ups(self, y_true, y_pred, class_true, class_pred):
        gen_true, gen_pred, gen_true_p, gen_pred_p = [], [], [], []
        for idx, (yt, yp) in enumerate(zip(class_true, class_pred)):
            if yt not in self.accuracy_targets:
                gen_true_p.append(y_true[idx])
                gen_pred_p.append(y_pred[idx])
                if yp not in self.accuracy_targets:
                    gen_true.append(y_true[idx])
                    gen_pred.append(y_pred[idx])
        return gen_true, gen_pred, gen_true_p, gen_pred_p

    def evaluate(self, y_true, y_pred):

        # Classification
        class_true = self.replace_follow_up_with_more(y_true)
        class_pred = self.replace_follow_up_with_more(y_pred)
        assert len(class_true) == len(y_true)
        assert len(class_pred) == len(y_pred)
        results = self.classification_evaluator.evaluate(class_true, class_pred)

        # Follow Up Generation
        # num_true_follow_ups = len([y_t for y_t in y_true if y_t.lower() not in self.labels])
        # num_pred_follow_ups = len([y_p for y_p in y_pred if y_p.lower() not in self.labels])
        # print(f'{num_true_follow_ups} follow-ups in ground truth. {num_pred_follow_ups} follow-ups predicted | {len(generation_y_true)} follow-up questions used for BLEU evaluation.')

        gen_y_true, gen_y_pred, gen_y_true_p, gen_y_pred_p = self.extract_follow_ups(
            y_true, y_pred, class_true, class_pred
        )
        assert len(gen_y_true) == len(gen_y_pred)
        assert len(gen_y_true_p) == len(gen_y_pred_p)
        results.update(
            self.more_evaluator.evaluate(gen_y_true, gen_y_pred, prefix="bleu")
        )
        results.update(
            self.more_evaluator.evaluate(gen_y_true_p, gen_y_pred_p, prefix="bleup")
        )
        results["combined"] = results["bleu4"] * results["macro"]
        results["num_bleu"] = len(gen_y_pred)
        results["num_bleup"] = len(gen_y_pred_p)
        results["num_total"] = len(y_pred)
        order = [
            "micro",
            "macro",
            "bleu1",
            "bleu4",
            "bleup1",
            "bleup4",
            "combined",
            "num_bleu",
            "num_total",
            "num_bleup",
            "bleu2",
            "bleu3",
            "bleup2",
            "bleup3",
        ]
        return {key: results[key] for key in order}


def prepro(text):
    doc = nlp(text, disable=["parser", "tagger", "ner"])
    result = ""
    for token in doc:
        orth = token.text
        if orth == "":
            result += " "
        elif orth == " ":
            result += " "
        else:
            result += orth.lower() + " "
    return result.strip().replace("\n", "")


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (
                possible_matches_by_order[i] + 1.0
            )
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (
                    float(matches_by_order[i]) / possible_matches_by_order[i]
                )
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def fetch_results(mode, ground_truths, predictions):
    if mode == "follow_ups":
        evaluator = MoreEvaluator()
        results = evaluator.evaluate(ground_truths, predictions)

    elif mode == "classification":
        evaluator = ClassificationEvaluator(labels=["yes", "no", "more", "irrelevant"])
        results = evaluator.evaluate(ground_truths, predictions)

    elif mode == "combined":
        evaluator = CombinedEvaluator(labels=["yes", "no", "more", "irrelevant"])
        results = evaluator.evaluate(ground_truths, predictions)

    return results


def preprocess_inputs(gold_file, pred_file):
    with open(gold_file, "r") as f:
        ground_truths = json.load(f)
    with open(pred_file, "r") as f:
        predictions = json.load(f)

    # Check if all IDs are aligned
    assert len(ground_truths) == len(predictions), "Predictions and ground truths have different sample sizes"

    ground_truth_map = {g["utterance_id"]: g for g in ground_truths}
    predictions_map = {p["utterance_id"]: p for p in predictions}

    for gid in ground_truth_map:
        assert gid in predictions_map

    # Extract answers and prepro
    ground_truths, predictions = [], []

    for uid in ground_truth_map.keys():
        instance = ground_truth_map[uid]
        if "all_answers" in instance:
            ground_truth = [prepro(answer) for answer in instance["all_answers"]]
        else:
            ground_truth = [prepro(instance["answer"])]
        prediction = prepro(predictions_map[uid]["answer"])
        ground_truths.append(ground_truth)
        predictions.append(prediction)

    return ground_truths, predictions


def evaluate(pred, gold=None, mode="combined"):
    assert mode in ["combined", "follow_ups", "classification"], "Mode not recognised"

    ground_truths, predictions = preprocess_inputs(gold, pred)
    results = fetch_results(mode, ground_truths, predictions)

    return results


def process_results(results, out):
    # Store results to disk
    with open(out, "w") as f:
        json.dump(results, f)

    # Print results
    try:
        from tabulate import tabulate

        tabulated = defaultdict(list)
        for method, scores in results.items():
            tabulated["method"].append(method)
            for metric, score in scores.items():
                tabulated[metric].append(score)
        print(tabulate(tabulated, headers="keys", tablefmt="presto"))
    except:
        print(json.dumps(results, indent=2))


def main(pred, gold=None, gold_multi=None, mode="combined", out=None):

    results = {}

    if gold:
        ground_truths, predictions = preprocess_inputs(gold, pred)
        results["gold"] = fetch_results(mode, ground_truths, predictions)

    if gold_multi:
        ground_truths, predictions = preprocess_inputs(gold_multi, pred)
        results["multi"] = fetch_results(mode, ground_truths, predictions)

    process_results(results, out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to predictions file")
    parser.add_argument("--gold", help="Path to gold file")
    parser.add_argument("--gold_multi", help="Path to multiple reference file")
    parser.add_argument(
        "--mode",
        default="combined",
        help="Mode to evaluate in",
        choices=["follow_ups", "classification", "combined"],
    )
    parser.add_argument("--out", default=None, help="Path to store scores")
    args = parser.parse_args()

    if not args.out:
        args.out = os.path.join(os.path.dirname(args.pred), "scores.json")

    if not (args.gold or args.gold_multi):
        raise ValueError("Atlest one of gold or gold_multi is required.")

    main(**vars(args))
               