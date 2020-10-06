# Whole MILC: generalizing learned dynamics across tasks, datasets, and populations

https://arxiv.org/abs/2007.16041

Usman Mahmood, Mahfuz M. Rahman, Alex Fedorov, Noah Lewis, Zengin Fu, Vince D. Calhoun, Sergey M. Plis


#### Dependencies:
* [PyTorch](https://github.com/pytorch/pytorch)
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
* [Catalyst](https://github.com/catalyst-team/catalyst)

```bash
conda install pytorch torchvision -c pytorch
conda install sklearn
```

### Installation 

```bash
# PyTorch
conda install pytorch torchvision -c pytorch
git clone https://github.com/UsmanMahmood27/MILC.git
cd MILC
pip install -e .
pip install -r requirements.txt
```

### MILC downstream:
Here's a sample example using pre-trained model for COBRE classification using 40 subjects per class for training: 

```bash
python -m scripts.run_ica_experiments_COBRE_catalyst --pre-training milc --script-ID 3 --exp UFPT --method sub-lstm 
```
Or run the jupyter notebook file with the same name. Requires installation of jupyter.
