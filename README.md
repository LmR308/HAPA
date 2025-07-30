# Dimension-Aware Active Annotation for Aesthetic Perception via Multi-Agent Human–AI Collaboration

This repository provides a PyTorch implementation of the Dimension-Aware Active Annotation for Aesthetic Perception via Multi-Agent Human–AI Collaboration presented in our AAAI 2026 paper.

## Get Started

### Requirements and Installation

The require environments are in **environment.yaml**, you can run below command to install the environment:

```python
conda env create -f environment.yaml
```

## Usage

Train the model by running main.py directly. Remember to set the chosen dataset, model backbone and hyper-parameters in the script.Please use the following command to load demo data for testing whether the environment is successfully installed.

```python
python main.py --data_path ./data --singleCapacity 5 --ReplayBuffer_capacity 4 --min_size 3 --sample_size 3
```

optional arguments:  
```--help``` show this help message and exit  
```--data_path``` show the the path of data  
```--singleCapacity``` show the size of a single batch of data  
```--ReplayBuffer_capacity``` show the capacity of ReplayBuffer in DQN  
```--min_size``` show the minimum data size for experience replay learning in DQN  
```--sample_size``` show the number of randomly sampled samples in DQN experience replay  

## Note

The data used in this experiment is anonymous and temporarily not publicly available, so encrypted data (data.json, reflect.json) is used.
