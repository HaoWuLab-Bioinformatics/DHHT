# DHHT

DHHT is a PyTorch implementation of **DHHT: Dynamic Hyperbolic Hypergraph Transformer for Graph Representation Learning**

Overview

!\[Dysformer Framework](https://github.com/HaoWuLab-Bioinformatics/DHHT/blob/master/framework.png)

## Installation

### 1\. Clone the repository

```bash
git clone https://github.com/HaoWuLab-Bioinformatics/DHHT.git
cd DHHT
```

### 2\. Create a virtual environment

Using Conda:

```bash
conda create -n DHHT python=3.9
conda activate DHHT
```

### 3\. Install dependencies

Install PyTorch according to your CUDA version from the official PyTorch website, then install the remaining packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric geoopt numpy scipy pandas scikit-learn openpyxl ogb tqdm
```

The main dependencies include:

```text
Python
PyTorch
PyTorch Geometric
Geoopt
NumPy
SciPy
Pandas
Scikit-learn
OpenPyXL
OGB
```

## Data Preparation

The repository contains adolescent health data under:

```text
data/zijian/jiaolvyiyu/
```

## Usage

### Medium-scale experiments

```bash
python yunxing\\\\\\\_medium.py
```



### Large-scale experiments

```
python yunxing\\\\\\\_large.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

