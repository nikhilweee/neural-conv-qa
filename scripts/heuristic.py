# Contains code from https://github.com/vzhong/e3/blob/0c6b771b27463427db274802c4417355ddd90ed7/preprocess_sharc.py

import editdistance
import os
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

import revtok
import string
import tempfile
from tqdm import tqdm

from evaluator_multi_pc import evaluate
# from evaluator import evaluate
import argparse


# Initialize Tokenizers

BERT_VOCAB = 'bert-base-uncased'
LOWERCASE = True
MATCH_IGNORE = {'do', 'have', '?'}
SPAN_IGNORE = set(string.punctuation)

bert_tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=LOWERCASE, cache_dir=None)


nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)


# Evaluation Functions

def evaluate_model(model_fn, dataset, multi=False, **kwargs):
    pred_json = []
    gold_json = []
    
    for utterance in dataset:
        pred_instance = {'utterance_id': utterance['utterance_id']}
        gold_instance = {'utterance_id': utterance['utterance_id']}

        if multi:
            gold_instance['all_answers'] = utterance['all_answers']
        else:
            gold_instance['answer'] = utterance['answer']

        pred_instance['answer'] = model_fn(utterance, **kwargs)
        pred_json.append(pred_instance)
        gold_json.append(gold_instance)

    with tempfile.NamedTemporaryFile('w', delete=False) as gold_file:
        json.dump(gold_json, gold_file)

    with tempfile.NamedTemporaryFile('w', delete=False) as pred_file:
        json.dump(pred_json, pred_file)

    return evaluate(gold=gold_file.name, pred=pred_file.name, mode='combined')


# Create Utterance Trees

def detokenize(tokens):
    words = []
    for i, t in enumerate(tokens):
        if t['orig_id'] is None or (i and t['orig_id'] == tokens[i-1]['orig_id']):
            continue
        else:
            words.append(t['orig'])
    return revtok.detokenize(words)


def filter_answer(answer):
    return detokenize([a for a in answer if a['orig'] not in MATCH_IGNORE])


def filter_chunk(answer):
    return detokenize([a for a in answer if a['orig'] not in MATCH_IGNORE])


def get_span(context, answer):
    answer = filter_answer(answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_chunk(context[i:j+1])
            if '\n' in chunk or '*' in chunk:
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j-i < best[1]-best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    s, e = best
    while not context[s]['orig'].strip() or context[s]['orig'] in SPAN_IGNORE:
        s += 1
    while not context[e]['orig'].strip() or context[s]['orig'] in SPAN_IGNORE:
        e -= 1
    return s, e


def get_bullets(context):
    indices = [i for i, c in enumerate(context) if c == '*']
    pairs = list(zip(indices, indices[1:] + [len(context)]))
    cleaned = []
    for s, e in pairs:
        while not context[e-1].strip():
            e -= 1
        while not context[s].strip() or context[s] == '*':
            s += 1
        if e - s > 2 and e - 2 < 45:
            cleaned.append((s, e-1))
    return cleaned


def subtokenize(doc):
    if not doc.strip():
        return []
    tokens = []
    for i, t in enumerate(revtok.tokenize(doc)):
        subtokens = bert_tokenizer.tokenize(t.strip())
        for st in subtokens:
            tokens.append({'orig': t, 'sub': st, 'orig_id': i})
    return tokens


def extract_clauses(data):
    snippet = data['snippet']
    t_snippet = subtokenize(snippet)
    questions = data['questions']
    t_questions = [subtokenize(q) for q in questions]

    spans = [get_span(t_snippet, q) for q in t_questions]
    bullets = get_bullets(t_snippet)
    all_spans = spans + bullets
    coverage = [False] * len(t_snippet)
    sorted_by_len = sorted(all_spans,  key=lambda tup: tup[1] - tup[0], reverse=True)

    ok = []
    for s, e in sorted_by_len:
        if not all(coverage[s:e+1]):
            for i in range(s, e+1):
                coverage[i] = True
            ok.append((s, e))
    ok.sort(key=lambda tup: tup[0])

    match = {}
    match_text = {}
    clauses = [None] * len(ok)
    for q, tq in zip(questions, t_questions):
        best_score = float('inf')
        best = None
        for i, (s, e) in enumerate(ok):
            score = editdistance.eval(detokenize(tq), detokenize(t_snippet[s:e+1]))
            if score < best_score:
                best_score, best = score, i
                clauses[i] = tq
        match[q] = best
        s, e = ok[best]
        match_text[q] = detokenize(t_snippet[s:e+1])

    clause_dict = {
        'questions': {q: tq for q, tq in zip(questions, t_questions)},
        'snippet': snippet, 't_snippet': t_snippet, 'spans': ok,
        'match': match, 'match_text': match_text, 'clauses': clauses
    }
    return clause_dict


def create_dev_trees(data):
    tasks = {}
    for ex in data:
        for h in ex['evidence']:
            if 'followup_question' in h:
                h['follow_up_question'] = h['followup_question']
                h['follow_up_answer'] = h['followup_answer']
                del h['followup_question']
                del h['followup_answer']
        if ex['tree_id'] in tasks:
            task = tasks[ex['tree_id']]
        else:
            task = tasks[ex['tree_id']] = {'snippet': ex['snippet'], 'questions': set()}
        for h in ex['history'] + ex['evidence']:
            task['questions'].add(h['follow_up_question'])
    keys = sorted(list(tasks.keys()))
    vals = [extract_clauses(tasks[k]) for k in tqdm(keys)]
    trees_dev = {k: v for k, v in zip(keys, vals)}
    return trees_dev


# Distribution based Model

def distribution_model(utterance):
    rule = utterance['snippet']
    history = utterance['history']
    scenario = utterance['scenario']
    question = utterance['question']
    
    
    turn_number = len(history) + 1
    
    if turn_number == 1:
        if history == [] and scenario == '':
            answer = 'Irrelevant'
        else:
            answer = rule
    else:
        answer = history[-1]['follow_up_answer']
        
    return answer


# Smart Model

def tokenize(text):
    return [token.text for token in spacy_tokenizer(text)]


def relevant_query(text, query, threshold=0.5):
    query_tokens = tokenize(query.lower())
    text_tokens = set(tokenize(text.lower()))
    
    relevant_tokens = 0
    total_tokens = 0
    
    for token in query_tokens:
        if token in STOP_WORDS or token in string.punctuation:
            continue
        elif token in text_tokens:
            relevant_tokens += 1
        total_tokens += 1
    
    return (relevant_tokens / total_tokens) >= threshold


def next_follow_up(utterance, trees_dev):
    previous_questions = set([x['follow_up_question'] for x in utterance['history']])
    
    tree = trees_dev[utterance['tree_id']]
    dic = {}
    for k, v in tree['match'].items():
        if v in dic:
            dic[v].add(k)
        else:
            dic[v] = {k}
    match = {tuple(v): tree['match_text'][list(v)[0]] for k, v in sorted(dic.items())}
    
    for questions_set, clause in match.items():
        if not any(question in previous_questions for question in questions_set):
            return 'Are you ' + clause + '?'
    return utterance['snippet']


def number_rules(rule):
    if '*' in rule: # bullet points
        return rule.count('*')
    else:
        return 1


def smart_model(utterance, trees_dev):
    rule = utterance['snippet']
    history = utterance['history']
    scenario = utterance['scenario']
    question = utterance['question']
    turn_number = len(history) + 1
    
    if turn_number == 1:
        if not scenario and not relevant_query(rule, question):
            return 'Irrelevant'
        else:
            return next_follow_up(utterance, trees_dev)
    elif turn_number == 2:
        if (not scenario and number_rules(rule) >= turn_number) or (scenario and number_rules(rule) - 1 >= turn_number):
            return next_follow_up(utterance, trees_dev)
        else:
            return history[-1]['follow_up_answer']
    else:
        return history[-1]['follow_up_answer']


def main(args):

    with open(args.dev_json, 'r') as f:
        data = json.load(f)

    if os.path.isfile(args.trees_json):
        with open(args.trees_json, 'r') as f:
            trees_dev = json.load(f)
    else:
        trees_dev = create_dev_trees(data)
        with open(args.trees_json, 'w') as f:
            json.dump(trees_dev, f)

    scores = evaluate_model(smart_model, data, multi=args.multi, trees_dev=trees_dev)

    results = {key: value for key, value in scores.items() if key in ['micro', 'macro', 'bleu4', 'bleup4']}

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A rule based heuristic model for ShARC.')
    parser.add_argument('dev_json', help='Path to the dev JSON file.')
    parser.add_argument('--multi', action='store_true', help='Whether to consider multiple references.')
    parser.add_argument('--trees-json', default='data/trees.json', help='path to utterance trees.')
    args = parser.parse_args()

    main(args)
