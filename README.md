# G2PA

## Train the Classifier

#### 1. Get [Biaobei Dataset](https://www.data-baker.com/open_source.html)

#### 2. Prepare for Aligner and do Alignment

- Run *pre.ipynb*
- Activate [MFA env](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
- Run `bash train_aligner.sh`
- Run `bash align.sh`
- Deactivate MFA env and run *aligned_pre.ipynb*
- Run `python hubert_extractor.py`
- Run `python resnet_trainer.py`
- To evaluate, run *evaluate_trainer.ipynb*


## Do Inference

- `cd test`
- Run `evaluate_g2p.ipynb` 
- 