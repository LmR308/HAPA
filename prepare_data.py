import torch
import json

def get_all_reflect_relation(data_path):
    """obtain drawing and course mapping relationships, dimensional correlation matrices
    Args:
        data_path (Str): the path where the data is stored
    Returns:
        Dict, Tensor(77,77): the dictionary stores the mapping of drawings and lessons, 
        Tensor(77, 77): dimension_adj_matrix stores 77 dimensions of interrelationships
    """    
    with open(f'{data_path}/reflect.json', 'r') as file:
        paint_to_course, dimension_adj_matrix = json.load(file)
    dimension_adj_matrix = torch.tensor(dimension_adj_matrix).cuda()
    return paint_to_course, dimension_adj_matrix


def get_RL_data(image_size, data_path):
    """preprocess the data and divide the training set and the test set
    Args:
        image_size (Int): image size
        data_path (Str): the path where the data is stored
    Returns:
        _type_: divide the training set and test set
    """    
    with open(f'{data_path}/data.json', 'r') as file:
        train_set, test_set = json.load(file)
    train_set = [[torch.tensor(_[0]).cuda(), _[1], _[2]] for _ in train_set]
    test_set = [[torch.tensor(_[0]).cuda(), _[1], _[2]] for _ in test_set]
    return train_set, test_set

def collate_RL(data):
    """merges a list of samples to form a mini-batch of Tensor(s).
    Args:
        data ([Tensor(), Str, List(num_classes)]): _description_
    Returns:
        Tensor(batch_size, 3, photo_size, photo_size): picture representation
        [Str]: the name of the picture
        Tensor(batch_size, num_classes): image tags
    """    
    img = [i[0].tolist() for i in data]
    name = [i[1] for i in data]
    labels = []
    for i in data:
        one_batch = []
        for j in i[-1]:
            one_batch.append(j)
        labels.append(one_batch)
    return torch.tensor(img), name, torch.tensor(labels)

def collate_data(data):
    img = torch.tensor([i[0].tolist() for i in data])
    name = [i[1] for i in data]
    labels = torch.tensor([i[-1].tolist() for i in data])
    return img, name, labels