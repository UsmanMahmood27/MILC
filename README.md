# Whole MILC: generalizing learned dynamics across tasks, datasets, and populations

Usman Mahmood*, Mahfuz M. Rahman*, Alex Fedorov*, Zengin Fu, Vince D. Calhoun, Sergey M. Plis





#### Dependencies:
* PyTorch
* Scikit-Learn
* Catalyst

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
Here's a sample example using pre-trained encoder for FBIRN classification: 

```bash
python -m scripts.run_probe --method infonce-stdim --env-name {env_name}
```
