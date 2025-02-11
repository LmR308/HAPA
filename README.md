# Empowering Aesthetic Education: Multi-Agent Active Learning for Human-AI Collaborative Painting Annotation

This repository provides a PyTorch implementation of the Human-AI collabo-rative Painting Annotation presented in our KDD 2025 paper.

## Get Started

### Requirements and Installation

The require environments are in **environment.yaml**, you can run below command to install the environment:

```python
conda env create -f environment.yaml
```

## Usage

Train the model by running the following command directly. Remember to set the chosen dataset, model backbone and hyper-parameters in the script.

```python
python RL_main.py --data_path ./data
```
