
# G2PA

## Install Requirements

`pip install requirements.txt`

## Train the Classifier

#### 1. Get [Biaobei Dataset](https://www.data-baker.com/open_source.html)

#### 2. Prepare for Aligner and do Alignment

- Run *pre.ipynb*
- Activate [MFA env](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
- `bash train_aligner.sh`
- `bash align.sh`
- Deactivate MFA env and run *aligned_pre.ipynb*
- `python hubert_extractor.py`
- `python resnet_trainer.py`
- To evaluate, run *evaluate_trainer.ipynb*


## Do Inference

- `cd test`
- Run **Prepare for G2PA**, the first block of *evaluate_g2p.ipynb*
- Activate MFA env and `bash align.sh`
- Deactivate MFA env and Run *aligned_pre.ipynb*
- Run *corrector.ipynb*
- Run **Evaluate**, the second block of *evaluate_g2p.ipynb*
