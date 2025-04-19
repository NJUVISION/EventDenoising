# EventDenoising

Official implementation of *"On Spatiotemporal Relation Modeling of Event: Parallel Denoising and No-Reference Evaluation"*.

## 🚀 Quick Start
### Environment
Python 3.10 \
Pytorch 2.2.1 \
CUDA 11.8 \
cudnn 8

### Installation
```bash
conda create -n EventDenoising python=3.10
conda activate EventDenoising
pip install -r requirements.txt
```
Require additional install [mamba-ssm==2.2.2](https://github.com/state-spaces/mamba) and [pytorch3d==0.7.5](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### EDformer++ Evaluation
```python
# Testing SNR with DVSCLEAN Dataset
python eval_DVSCLEAN.py
```
```python
# Testing AUC with DND21 Dataset
python eval_DND21.py
```
### ENRR Evaluation
```python
# Testing ENR with DND21 Dataset
python eval_enrr_DND21.py
```
```python
# Testing ENR with ED24 Dataset
python eval_enrr_ED24.py
```

## 🏋️ Training
```python
# EDfromer++ train
python train_edformer_plus.py

# ENRR train
python train_enrr.py
```

## 📦  Datasets
You can use the sample data in ./data for quick testing, or download the complete dataset from the link below for full evaluation or model training. 
| Dataset  | Usage | 
|----------| --------|
|[ED24](https://pan.baidu.com/s/18DWwTg7LNiuJN4dxTp51eg?pwd=3905) | EDformer++ and ENRR Training |
|[DND21](https://pan.baidu.com/s/1Z61K5hRxjxEV-Pfs28_htg?pwd=3905) | AUC and ENR Evaluation |
|[DVSCLEAN](https://drive.google.com/file/d/14FJD-kf9NA-bdWVWHK35ewLiLVdBnNSq/view?usp=share_link) | SNR Evaluation |
|[ED-KoGTL](https://github.com/yusra-alkendi/ed-kogtl) | SNR Evaluation |
|[E-MLB](https://github.com/KugaMaxx/cuke-emlb) | MESR Evaluation |

## ⁉️ Contact
For technical issues, please contact:
📧 jiangbin@smail.nju.edu.cn
