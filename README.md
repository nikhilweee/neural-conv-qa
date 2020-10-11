# Neural Conversational QA
This repository contains the script and data used for the EMNLP 2020 paper titled **Neural Conversational QA: Learning to Reason vs Exploiting Patterns**.


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
@misc{sharma2019neural,
      title={Neural Conversational QA: Learning to Reason v.s. Exploiting Patterns}, 
      author={Abhishek Sharma and Danish Contractor and Harshit Kumar and Sachindra Joshi},
      year={2019},
      eprint={1909.03759},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```