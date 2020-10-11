import json
import spacy
import regex
import argparse
from difflib import SequenceMatcher
from tqdm import trange
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])


def find_prefix_length(text, subtext):
    """Find the length of the prefix after which a subtext occurs in the given text."""

    for idx in range(len(text)):
        if text[idx:idx+len(subtext)] == subtext:
            return idx
    return -1


def all_stop_words(token_list, span):
    """Checks whether a span in a list of tokens only contains stop words."""

    for i in range(span[0], span[1] + 1):
        if token_list[i].lower() not in STOP_WORDS:
            return False
    else:
        return True


def find_closest_element_ix(offsets, number):
    """Returns the index of closest element in sorted list."""
    for i, element in enumerate(offsets):
        if element == number:
            return i
        elif element > number:
            break
    just_larger_ix = i
    if just_larger_ix == 0 or offsets[just_larger_ix] - number < number - offsets[just_larger_ix - 1]:
        return just_larger_ix
    else:
        return just_larger_ix - 1


def is_follow_up(text):
    """Checks whether an answer is a follow-up question."""

    if text.lower() not in ['yes', 'no', 'irrelevant']:
        return True
    else:
        return False


def split_tokens(tokens, delimiters):
    """Split a list of tokens on a list of delimiters."""

    delim_indices = [token[0] + 1 for token in tokens if token[1] in delimiters]

    start_indices = [0] + delim_indices

    if len(delim_indices) == 0 or delim_indices[-1] != len(tokens):
        end_indices = delim_indices + [len(tokens)]
    else:
        end_indices = delim_indices

    splits = [tokens[start: end] for start, end in zip(start_indices, end_indices)]
    return splits


def get_clause_indices(text):
    """Return all clause indices which start with an asterisk (*)."""

    tokens = list(enumerate([token.text.lower() for token in nlp(text)]))
    splits = split_tokens(tokens, ['\n', '\n\n'])

    indices = []
    for split in splits:
        if split[0][1] == '*':
            indices.append((split[0][0], split[-1][0]))
    return indices


def check_intersects(s1, s2):
    """Check whether the spans s1 and s2 intersect."""

    if (s2[0] <= s1[0] and s2[1] >= s1[0]) or (s1[0] <= s2[0] and s1[1] >= s2[0]):
        return True
    else:
        return False


def find_lcs(text1, text2, tokenizer_fn, min_length=3, fuzzy_matching=True, filter_stop_words=True):
    """
    Returns start and end (token) index of longest common subsequence (of text1 and text2) in text1.
    If fuzzy matching is True, it is used when lcs is less than min_length.
    If filter_stop_words is True, in case the found span contains only stop words, None is returned. 
    """

    text1_tokens = [token.text.lower() for token in tokenizer_fn(text1)]
    text1_offsets = [(token.idx, token.idx + len(token.text))
                     for token in tokenizer_fn(text1)]
    text2_tokens = [token.text.lower() for token in tokenizer_fn(text2)]

    sequence_matcher = SequenceMatcher(
        None, text1_tokens, text2_tokens, autojunk=False)
    lcs_match = sequence_matcher.find_longest_match(
        0, len(text1_tokens), 0, len(text2_tokens))
    lcs_span = lcs_match.a, lcs_match.a + lcs_match.size - 1
    regex_span = None

    if (lcs_match.size < min_length or all_stop_words(text1_tokens, lcs_span)) and fuzzy_matching:
        pattern = r'(?:\b' + regex.escape(text2.lower()) + r'\b){i<=6,d<=20}'
        # This can take quite some time.
        regex_match = regex.search(pattern, text1.lower(), regex.BESTMATCH)
        if regex_match:
            start_token_ix = find_closest_element_ix(
                [offset[0] for offset in text1_offsets], regex_match.span()[0])
            end_token_ix = find_closest_element_ix(
                [offset[1] for offset in text1_offsets], regex_match.span()[1])
            regex_span = start_token_ix, end_token_ix

    if regex_span is not None and not (filter_stop_words and all_stop_words(text1_tokens, regex_span)):
        match = regex_span
    elif lcs_match.size > 0 and not (filter_stop_words and all_stop_words(text1_tokens, lcs_span)):
        match = lcs_span
    else:
        match = None

    return match


def add_answers(dataset):
    """Adds new clauses as an additional reference answer."""

    for idx in trange(len(dataset)):
        snippet = dataset[idx]['snippet']
        answer = dataset[idx]['answer']
        answers = [answer]
        follow_up_questions = [item['follow_up_question'] for item in dataset[idx]['history']]

        # Extract asterisk-based (*) clauses from rule.
        rule_clauses = get_clause_indices(snippet)

        if len(rule_clauses) >= 1 and is_follow_up(answer):

            # Extract clause spans from follow-up questions
            question_clauses = []
            for question in follow_up_questions:
                matched_clause = find_lcs(snippet, question, nlp)
                if matched_clause is not None:
                    question_clauses.append(matched_clause)

            # Find intersecting question-clauses
            match_filter = [False] * len(rule_clauses)
            for c, rule_clause in enumerate(rule_clauses):
                for question_clause in question_clauses:
                    if check_intersects(question_clause, rule_clause):
                        match_filter[c] = True

            # Find clauses corresponding to a the answer
            answer_clause = find_lcs(snippet, answer, nlp)
            if answer_clause is not None:
                is_asterisk_clause = False

                # Find intersecting answer-clauses
                for c, rule_clause in enumerate(rule_clauses):
                    if not match_filter[c]:
                        if check_intersects(answer_clause, rule_clause):
                            is_asterisk_clause = True
                            match_filter[c] = True

                if is_asterisk_clause:
                    # Separate the prefix and suffix
                    answer_clause_span = nlp(snippet)[answer_clause[0]:answer_clause[1]+1]
                    answer_clause_tokens = [token.text.lower() for token in answer_clause_span]
                    answer_tokens = [token.text.lower() for token in nlp(answer)]
                    prefix_length = find_prefix_length(answer_tokens, answer_clause_tokens)
                    suffix_length = len(answer_tokens) - len(answer_clause_tokens) - prefix_length

                    for c, rule_clause in enumerate(rule_clauses):
                        # Add rule clauses which have not yet occured in the history
                        if not match_filter[c]:
                            rule_clause_span = nlp(snippet)[rule_clause[0]:rule_clause[1]+1][1:]
                            rule_clause_tokens = [token.text.lower() for token in rule_clause_span]
                            new_answer_tokens = answer_tokens[:prefix_length] + rule_clause_tokens + answer_tokens[-suffix_length:]
                            answers.append(' '.join(new_answer_tokens))

        dataset[idx]['all_answers'] = answers
    return dataset

def main(args):
    with open(args.dev_in, 'r') as f:
        dataset = json.load(f)
    dataset = add_answers(dataset)
    with open(args.dev_out, 'w') as f:
        json.dump(dataset, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-in", required=True, help="path to dev input file")
    parser.add_argument("--dev-out", required=True, help="path to dev output file")
    args = parser.parse_args()

    main(args)
