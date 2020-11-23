# Neural Conversational QA
This repository contains the script and data used for the EMNLP 2020 paper titled **Neural Conversational QA: Learning to Reason vs Exploiting Patterns** [[ArXiV]](https://arxiv.org/abs/1909.03759) [[ACL]](https://www.aclweb.org/anthology/2020.emnlp-main.589/).


## Repository structure
The repository is arranged as follows. The datasets folder contains all datasets used in the paper. The file names are of the form `<dataset>_<suffix>.json` where `dataset` may be one of `history`, `mod` or `sharc` representing the History-Shuffled dataset (Section 4), the ShARC-Mod dataset (Section 5) or the official ShARC datasets as mentioned in the paper. `suffix` may be one of `train`, `dev` and `dev_multi` representing the training set, the dev set, and the dev set with multiple references (Section 3, Appendix A.3).

```
.
├── README.md
├── datasets                            # all datasets used in the paper
├── requirements.txt
├── run.sh                              # wrapper script to recreate datasets
├── scripts                             # scripts to create modified datasets
│   ├── create_history_shuffled.py
│   ├── create_multi_ref.py
│   ├── create_sharc_mod.py
│   ├── evaluator_multi_pc.py
│   └── heuristic.py                    # the heuristic based program (Section 2.1)
└── sharc1-official                     # the official ShARC dataset
```

The scripts folder has scripts to recreate the datasets, as well as the heuristic based program `heuristics.py` (Section 2.1). `sharc1-official` is directly downloaded from https://sharc-data.github.io/data.html, and `requirements.txt` has all dependencies necessary to run the scripts.


## Recreating datasets

if you wish to use the scripts to recreate the datasets used in the paper, the `run.sh` script should be useful. Please make sure you use the correct version of the dependencies. Not doing so might lead to different resutls. We recommend that you use conda as described below.

```
$ conda create -n py36 python=3.6
$ conda activate py36
$ pip install -r requirements.txt
$ python3 -m spacy download en_core_web_sm
$ python3 -m spacy download en_core_web_md
$ bash run.sh
```

## Citation

If you find this repository useful, kindly use the following.

```
@inproceedings{verma-etal-2020-neural,
    title = "Neural Conversational {QA}: Learning to Reason vs Exploiting Patterns",
    author = "Verma, Nikhil  and
      Sharma, Abhishek  and
      Madan, Dhiraj  and
      Contractor, Danish  and
      Kumar, Harshit  and
      Joshi, Sachindra",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.589",
    pages = "7263--7269",
    abstract = "Neural Conversational QA tasks such as ShARC require systems to answer questions based on the contents of a given passage. On studying recent state-of-the-art models on the ShARC QA task, we found indications that the model(s) learn spurious clues/patterns in the data-set. Further, a heuristic-based program, built to exploit these patterns, had comparative performance to that of the neural models. In this paper we share our findings about the four types of patterns in the ShARC corpus and how the neural models exploit them. Motivated by the above findings, we create and share a modified data-set that has fewer spurious patterns than the original data-set, consequently allowing models to learn better.",
}
```

## Acknowledgements
Our heuristic model `scripts/heuristic.py` contains code derived from the [preprocessing script](https://github.com/vzhong/e3/blob/0c6b771b27463427db274802c4417355ddd90ed7/preprocess_sharc.py) in https://github.com/vzhong/e3. The evaluation script `scripts/evaluator_multi_pc.py` is derived from the [official evaluation script](https://worksheets.codalab.org/worksheets/0xcd87fe339fa2493aac9396a3a27bbae8/)
