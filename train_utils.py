import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models import MLP

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.functional import F
from ucimlrepo import fetch_ucirepo # TODO: pip install ucimlrepo


class TabDataset(TensorDataset):
    def __init__(self, *tensors, **kwargs):
        super().__init__(*tensors, **kwargs)
        self.targets = tensors[1]

def get_optimizer(model_parameters, **kwargs):
    """
    Returns an optimizer based on the provided configuration.
    
    Args:
        model_parameters: Parameters of the model to optimize.
        **kwargs: Keyword arguments from configuration.
                  Expected keys:
                  - name (str): Name of the optimizer ('sgd', 'adam', 'adamw', 'rmsprop', etc.).
                  - Any other key-value pairs compatible with the optimizer's arguments.
    
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    
    optimizer_name = kwargs.pop('name', 'Adam') # this sets adam as default optimizer
    
    if hasattr(optim, optimizer_name):
        optimizer_class = getattr(optim, optimizer_name)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")
    
    return optimizer_class(model_parameters, **kwargs)


def confusion_matrix(predictions: torch.tensor, labels: torch.tensor, num_classes: int):
    """
    Compute the confusion matrix using PyTorch.

    Args:
    - predictions: Tensor of shape (n_samples,) with predicted class indices.
    - labels: Tensor of shape (n_samples,) with true class indices.
    - num_classes: Number of classes.

    Returns:
    - conf_matrix: Tensor of shape (num_classes, num_classes) with the confusion matrix.
    """
    # Initialize the confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Populate the confusion matrix
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix


class ClassesList:
    """
    A class to handle a list of classes for (multiple) classification tasks.
    """
    def __init__(self, classes:list, dataset_name=None, sort=True):
        if isinstance(classes, ClassesList):
            assert dataset_name == classes.dataset_name, "Something weird happened, dataset_name should be the same."
            classes = classes.classes

        self.is_multidataset = isinstance(classes[0], list) or isinstance(classes[0], tuple)

        if sort:
            if self.is_multidataset:
                for cl_list in classes:
                    cl_list.sort()
            else:
                classes.sort()

        if self.is_multidataset:
            assert dataset_name is not None, "dataset_name must be provided for multiple datasets."
            assert len(classes) == len(dataset_name), "Number of classes must match number of datasets."
            self._items = [(ds, cl) for ds, ds_cls in zip(dataset_name, classes) for cl in ds_cls]
            
            self._ordinal_mapping = {}
            # for i, cl in enumerate(classes):
            #     for j, c in enumerate(cl):
            #         # if not relabeled, use the original class index c instead of j
            #         self.ordinal_mapping[j, i] = sum(len(classes[l]) for l in range(i)) + j
            for i, (ds, cl) in enumerate(self._items):
                ds_idx = dataset_name.index(ds)
                self._ordinal_mapping[cl, ds_idx] = i
        else:
            if isinstance(dataset_name, list) or isinstance(dataset_name, tuple):
                dataset_name = dataset_name[0]
            self._items = [(dataset_name, cl) for cl in classes]
        
        self.dataset_name = dataset_name
        self.num_classes = len(classes) if not self.is_multidataset else sum(len(c) for c in classes)

        self.classes = classes
        self.labels = [f"{ds}_{cl}" for ds, cl in self._items] if self.is_multidataset else classes

    def classes_perds(self):
        if self.is_multidataset:
            return [(self.dataset_name[i], cl) for i, cl in enumerate(self.classes)]
        else:
            return [(self.dataset_name, self.classes)]
    
    def get_dataset(self, class_id):
        if isinstance(class_id, str):
            class_id = int(class_id)
        if isinstance(class_id, int):
            class_id = [class_id]
        datasets = [self._items[i][0] for i in class_id]
        assert len(set(datasets)) == 1, f"Class {class_id} belongs to multiple datasets: {datasets}"
        return datasets[0]
    
    def get_dataset_name(self, index):
        if isinstance(index, str):
            index = int(index)
        return self._items[index][0]
        
    def __len__(self):
        return self.num_classes
    
    def __getitem__(self, index):
        return self._items[index][1]
        
    def __repr__(self):
        return f"ClassesList({self.classes}, dataset_name={self.dataset_name})"

    def __eq__(self, other):
        # Equality with list returning True if the the list is equal to the classes
        if type(other) in [list, ClassesList]:
            other_classes = other if isinstance(other, list) else other.classes
            if self.is_multidataset:
                return all([self.classes[i] == other_classes[i] for i in range(len(self.classes))])
            else:
                return self.classes == other_classes
        else:
            return False
        
    def __ne__(self, other):
        return not self.__eq__(other)

uci_ids = {
    'abalone': 1, # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    'covertype': 31, # [0, 1, 2]
    'letter_recognition':59, # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    'yeast': 110, # [0, 1, 2, 3]
    'human_activity_recognition': 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip',
}

def get_tabular_loaders(ds_id, classes: ClassesList, batch_size: int = 64, no_relabel: bool = False, return_dataset=False):
    if isinstance(ds_id, int):
        dataset = fetch_ucirepo(id=ds_id)
        X = dataset.data.features
        y = dataset.data.targets.iloc[:, 0]
        # remove missing values
        X = X.dropna(axis=1, how='any')

        # remove classes appearing rarely
        counts = y.value_counts(normalize=True)
        mask = counts[counts >= y.nunique()**-1 * .4].index
        y = y[y.isin(mask)]
        X = X.loc[y.index]
        y = y.astype('category').cat.codes

        # Get one-hot encoding for categorical features
        for col in X.select_dtypes(exclude=['number']).columns:
            if X[col].nunique() < 10:
                X = pd.get_dummies(X, columns=[col], drop_first=True, dtype=int)
            else:
                X[col] = X[col].astype('category').cat.codes
        # assert all features are numeric
        assert X.select_dtypes(include=['number']).shape[1] == X.shape[1], f"Non-numeric features found ({X.select_dtypes(exclude=['number']).columns})"

        # Normalize numeric features
        scaler = StandardScaler()
        X[X.columns] = scaler.fit_transform(X[X.columns])

        # remove features with low variance
        X = X.loc[:, (X.var() > 0.01) & (X.nunique() > 1)]
        y = y.astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    elif isinstance(ds_id, str) and 'human+activity' in ds_id:
        X_train, X_test, y_train, y_test = get_human_activity_loaders(ds_id)
    else:
        raise ValueError(f"Dataset {ds_id} not supported.")

    assert np.allclose(sorted(y_train.unique()), sorted(y_test.unique()))

    train_dataset = TabDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TabDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

    # Filter the dataset to only include the specified classes
    train_indices = torch.isin(train_dataset.tensors[1], torch.tensor(classes))
    test_indices = torch.isin(test_dataset.tensors[1], torch.tensor(classes))

    train_dataset = Subset(train_dataset, torch.nonzero(train_indices).squeeze())
    test_dataset = Subset(test_dataset, torch.nonzero(test_indices).squeeze())

    label_mapping = {original: new for new, original in enumerate(classes)}

    # Relabel the targets in the train and test subsets
    def relabel_dataset(subset, original_targets):
        for idx in range(len(subset)):
            original_label = original_targets[subset.indices[idx]]
            new_label = label_mapping[original_label.item()]
            subset.dataset.tensors[1][subset.indices[idx]] = new_label
    if not no_relabel:
        relabel_dataset(train_dataset, train_dataset.dataset.tensors[1])
        relabel_dataset(test_dataset, test_dataset.dataset.tensors[1])

    # Split train_dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_human_activity_loaders(ds_id):
    ds_path = Path('data', ds_id.split('/')[-1])
    if not ds_path.with_suffix('').exists() or len(list(ds_path.with_suffix('').iterdir())) == 0:
        # Download the dataset
        import wget
        import zipfile
        import shutil
        print(f"Downloading dataset from {ds_id}...")
        wget.download(ds_id, out=str(ds_path))
        # Unzip the dataset
        with zipfile.ZipFile(str(ds_path), 'r') as zip_ref:
            zip_ref.extractall(str(ds_path.with_suffix('')))
        zip_content_path = next(ds_path.with_suffix('').glob('*.zip'))
        with zipfile.ZipFile(str(zip_content_path), 'r') as zip_content:
            zip_content.extractall(str(ds_path.with_suffix('')))
        # Move the extracted files to the correct directory
        for item in ds_path.iterdir():
            if item.is_dir():
                for sub_item in item.iterdir():
                    if sub_item.is_file():
                        shutil.move(str(sub_item), str(ds_path))
                shutil.rmtree(str(item))
            else:
                shutil.move(str(item), str(ds_path))
    # Load the dataset
    ds_path = ds_path.with_suffix('') / 'UCI HAR Dataset'
    
    # feature_names = pd.read_csv(ds_path / 'features.txt', delim_whitespace=True, header=None)
    # feature_names = feature_names.iloc[:, 1].values
    X_train = pd.read_csv(ds_path / 'train' / 'X_train.txt', sep=r'\s+', header=None)#, names=feature_names)
    y_train = pd.read_csv(ds_path / 'train' / 'y_train.txt', sep=r'\s+', header=None).iloc[:, 0]
    X_test = pd.read_csv(ds_path / 'test' / 'X_test.txt', sep=r'\s+', header=None)#, names=feature_names)
    y_test = pd.read_csv(ds_path / 'test' / 'y_test.txt', sep=r'\s+', header=None).iloc[:, 0]
    y_train -= 1 # classes are 1-indexed
    y_test -= 1
    
    # Normalize numeric features
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    
    return X_train, X_test, y_train, y_test


def flatten_dataset(dataset, verbose=False):
    flatten_x = []
    flatten_y = []
    it = tqdm(dataset, desc="Flattening dataset") if verbose else dataset
    for x, y in it:
        flatten_x.append(x.flatten())
        flatten_y.append(y)
    flatten_x = torch.stack(flatten_x)
    flatten_y = torch.tensor(flatten_y)
    return TabDataset(flatten_x, flatten_y)


def get_fashionmnist_loaders(classes: list, batch_size: int = 64, no_relabel: bool = False, flatten: bool = False, return_dataset=False):
    """
    Returns data loaders for the FashionMNIST dataset filtered to only include the specified classes.

    Args:
        classes (list of int): List of classes to include in the dataset.
        batch_size (int): Batch size for the data loaders.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    torch.manual_seed(0)  # Ensure reproducibility
    
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        
    # Filter the dataset to only include the specified classes
    train_indices = torch.isin(train_dataset.targets, torch.tensor(classes))
    test_indices = torch.isin(test_dataset.targets, torch.tensor(classes))
    
        
    train_dataset = Subset(train_dataset, torch.nonzero(train_indices).squeeze())
    test_dataset = Subset(test_dataset, torch.nonzero(test_indices).squeeze())

    label_mapping = {original: new for new, original in enumerate(classes)}
    
    # Relabel the targets in the train and test subsets
    def relabel_dataset(subset, original_targets):
        for idx in range(len(subset)):
            original_label = original_targets[subset.indices[idx]]
            new_label = label_mapping[original_label.item()]
            subset.dataset.targets[subset.indices[idx]] = new_label

    if not no_relabel:
        relabel_dataset(train_dataset, train_dataset.dataset.targets)
        relabel_dataset(test_dataset, test_dataset.dataset.targets)

    if flatten:
        train_dataset = flatten_dataset(train_dataset)
        test_dataset = flatten_dataset(test_dataset)

    # Split train_dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    if return_dataset:
        return train_subset, val_subset, test_dataset

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_cifar10_loaders(classes: list, batch_size: int = 64, no_relabel: bool = False, return_dataset=False):
    """
    Returns data loaders for the CIFAR-10 dataset converted to grayscale 
    and filtered to only include the specified classes.

    Args:
        classes (list of int): List of classes to include in the dataset.
        batch_size (int): Batch size for the data loaders.
        no_relabel (bool): Just for compatibility with the other datasets.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    torch.manual_seed(0)  # Per riproducibilità

    # Trasformazioni per CIFAR-10 (conversione in scala di grigi + normalizzazione)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Converti in scala di grigi (1 canale)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizzazione su 1 canale
    ])

    # Caricamento dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Filtraggio delle classi desiderate
    train_indices = torch.isin(torch.tensor(train_dataset.targets), torch.tensor(classes))
    test_indices = torch.isin(torch.tensor(test_dataset.targets), torch.tensor(classes))

    train_dataset = Subset(train_dataset, torch.nonzero(train_indices).squeeze())
    test_dataset = Subset(test_dataset, torch.nonzero(test_indices).squeeze())

    # Mappatura delle etichette (es. [2, 5, 9] → [0, 1, 2])
    label_mapping = {original: new for new, original in enumerate(classes)}

    def relabel_dataset(subset, original_targets):
        for idx in range(len(subset)):
            original_label = original_targets[subset.indices[idx]]
            new_label = label_mapping[original_label]
            subset.dataset.targets[subset.indices[idx]] = new_label

    relabel_dataset(train_dataset, train_dataset.dataset.targets)
    relabel_dataset(test_dataset, test_dataset.dataset.targets)

    # Suddivisione in training (80%) e validation (20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    if return_dataset:
        return train_subset, val_subset, test_dataset

    # Creazione DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_mnist_loaders(classes: list, batch_size: int = 64, no_relabel: bool = False, flatten: bool = False, return_dataset=False):
    """
    Returns data loaders for the MNIST dataset filtered to only include the specified classes.

    Args:
        classes (list of int): List of classes to include in the dataset.
        batch_size (int): Batch size for the data loaders.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    torch.manual_seed(0) # Important! This is in order to be sure that training the models 
                         # and testing in the notebook report the results according to the same images
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if flatten:
        train_dataset = flatten_dataset(train_dataset)
        test_dataset = flatten_dataset(test_dataset)
        
    # Filter the dataset to only include the specified classes
    train_indices = torch.isin(train_dataset.targets, torch.tensor(classes))
    test_indices = torch.isin(test_dataset.targets, torch.tensor(classes))
    
    train_dataset = Subset(train_dataset, torch.nonzero(train_indices).squeeze())
    test_dataset = Subset(test_dataset, torch.nonzero(test_indices).squeeze())

    label_mapping = {original: new for new, original in enumerate(classes)}
    
    # Relabel the targets in the train and test subsets
    def relabel_dataset(subset, original_targets):
        for idx in range(len(subset)):
            original_label = original_targets[subset.indices[idx]]
            new_label = label_mapping[original_label.item()]
            subset.dataset.targets[subset.indices[idx]] = new_label

    if not no_relabel:
        relabel_dataset(train_dataset, train_dataset.dataset.targets)
        relabel_dataset(test_dataset, test_dataset.dataset.targets)

    # Split train_dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    if return_dataset:
        return train_subset, val_subset, test_dataset

    # Create data loaders
    # if test_batch_size is None:
    #     test_batch_size = len(test_dataset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_loaders(classes, batch_size: int = 64, no_relabel: bool = False, flatten=False, return_dataset=False):
    if classes.is_multidataset:
        # mixed dataset
        return get_mixed_loaders(classes, batch_size, no_relabel, flatten=flatten, return_dataset=return_dataset)
    elif classes.dataset_name == 'mnist':
        return get_mnist_loaders(classes, batch_size, no_relabel, flatten=flatten, return_dataset=return_dataset) # bisogna fare un ciclo sulle classes
    elif classes.dataset_name == 'cifar10':
        return get_cifar10_loaders(classes, batch_size, no_relabel) # bisogna fare un ciclo sulle classes
    elif classes.dataset_name == 'fashion_mnist':
        return get_fashionmnist_loaders(classes, batch_size, no_relabel, flatten=flatten, return_dataset=return_dataset)
    elif classes.dataset_name in uci_ids:
        return get_tabular_loaders(uci_ids[classes.dataset_name], classes, batch_size, no_relabel, return_dataset=return_dataset)
    else:
        raise ValueError(f"Dataset {classes.dataset_name} not supported.")


def get_mixed_loaders(classes: ClassesList, batch_size: int = 64, no_relabel: bool = False, flatten=True, return_dataset=False):
    """
    Returns data loaders for the specified datasets with the specified classes for each of them.

    Args:
        classes (list of list of int): List of classes for each dataset.
        datasets (tuple): Tuple of dataset names (e.g., ('mnist', 'cifar10')).
        batch_size (int): Batch size for the data loaders.
        no_relabel (bool): If True, do not relabel the classes in the datasets.
    Returns:
        loaders (list of DataLoader): List of DataLoaders for each dataset.
    """
    assert len(classes) == 10, "Total number of classes must be 10."
    assert no_relabel == False, "no_relabel with mixed datasets do not preserve differences between the datasets."
    
    dstrain_all = []
    dsval_all = []
    dstest_all = []
    label_mapping = classes._ordinal_mapping

    def relabel_dataset(original_targets, subset_indices, ds_idx):
        this_mapping = torch.tensor([label_mapping.get((cl.item(), ds_idx), -1) for cl in original_targets.unique()]) # return the mapping for the current dataset, None if the label is not in the mapping
        new_labels = this_mapping[original_targets[subset_indices]]

        assert (new_labels == -1).sum() == 0, f"Some labels are not in the mapping: {new_labels[new_labels == -1]}"
        original_targets[subset_indices] = new_labels

    for i, (dataset, class_list) in enumerate(classes.classes_perds()):
        clist = ClassesList(class_list, dataset_name=dataset)
        dstrain, dsval, dstest = get_loaders(clist, batch_size, flatten=flatten, return_dataset=True, no_relabel=True)
        relabel_dataset(dstrain.dataset.dataset.targets, dstrain.dataset.indices[dstrain.indices], i)
        relabel_dataset(dsval.dataset.dataset.targets, dsval.dataset.indices[dsval.indices], i)
        relabel_dataset(dstest.dataset.targets, dstest.indices, i)

        dstrain_all.append(dstrain)
        dsval_all.append(dsval)
        dstest_all.append(dstest)

    dstrain_all = ConcatDataset(dstrain_all)
    dsval_all = ConcatDataset(dsval_all)
    dstest_all = ConcatDataset(dstest_all)
    if return_dataset:
        return dstrain_all, dsval_all, dstest_all
    
    train_loader = DataLoader(dstrain_all, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dsval_all, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dstest_all, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main_train(seeds: list, 
               classes: list, 
               train_loader: DataLoader, 
               test_loader: DataLoader, 
               epochs: int = 10, 
               hidden_dims: list = [16, 8], 
               verbose: bool = False,
               activation_fn: nn.Module = nn.ReLU, 
               positivity_constraint: bool=False, 
               binary: bool = True):
    
    for seed in seeds:

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = MLP(hidden_dims = hidden_dims, output_dim=len(classes), activation_fn=activation_fn, positivity_constraint=positivity_constraint, binary=binary)
        model = MLP(hidden_dims = hidden_dims, output_dim=len(classes), activation_fn=activation_fn, positivity_constraint=positivity_constraint, binary=binary)
        # model = SimpleNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        str_classes = ''
        for i in classes:
            str_classes+=str(i)
        # Save the trained model
        os.makedirs(f'models_{str_classes}', exist_ok=True)
        os.makedirs(f'models_{str_classes}/initial_models', exist_ok=True)
        model_path = f'models_{str_classes}/initial_models/{seed}.pth'
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    },  model_path)
        torch.save(model.state_dict(), model_path)
        # print(f'Model saved to {model_path}')
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_labels = []
            total_pred = []
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # if positivity_constraint:
                #     # Clamping the weights to be non-negative
                #     with torch.no_grad():
                #         for param in model.parameters():
                #             param.clamp_(min=0)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_labels.append(labels)
                total_pred.append(predicted)
            total_pred = torch.cat(total_pred, dim = 0)
            total_labels = torch.cat(total_labels, dim = 0)
            accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy}")


        os.makedirs(f'models_{str_classes}', exist_ok=True)
        os.makedirs(f'models_{str_classes}/final_models', exist_ok=True)
        model_path = f'models_{str_classes}/final_models/{seed}.pth'
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)
        torch.save(model.state_dict(), model_path)
        
        accuracy, cm = eval(model, seed, test_loader)

        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return 

@torch.no_grad()
def eval(model, test_loader: DataLoader, rewarded_classes, n_classes:int = 10, device: str = 'cpu'):

    model.eval()
    total_labels = []
    total_pred = []
    rewarded_pred = []
    rewarded_labels = []
    non_rewarded_entropy = []
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

        entropy = -torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)  # entropy H(p)

        rewarded_mask = torch.tensor([label.item() in rewarded_classes for label in labels], dtype=torch.bool, device=labels.device)
        
        if hasattr(test_loader, 'label_mapping'):
            # Apply the label mapping to the predictions and labels
            # predicted = torch.tensor([test_loader.label_mapping[i.item(), test_loader.current_loader] for i in predicted])
            labels = torch.tensor([test_loader.label_mapping[i.item(), test_loader.current_loader] for i in labels])

        total_labels.append(labels)
        total_pred.append(predicted)

        rewarded_labels.append(labels[rewarded_mask])
        rewarded_pred.append(predicted[rewarded_mask])
        non_rewarded_entropy.append(entropy[~rewarded_mask])


    total_pred = torch.cat(total_pred, dim = 0)
    total_labels = torch.cat(total_labels, dim = 0)

    rewarded_pred = torch.cat(rewarded_pred, dim = 0)
    rewarded_labels = torch.cat(rewarded_labels, dim = 0)
    non_rewarded_entropy = torch.cat(non_rewarded_entropy, dim = 0)

    accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
    cm = confusion_matrix(total_pred, total_labels, total_labels.unique().size(0))

    rewarded_accuracy = (rewarded_labels == rewarded_pred).sum().item() / len(rewarded_labels)

    return accuracy, cm, rewarded_accuracy, torch.mean(non_rewarded_entropy).item()
        

@torch.no_grad()
def mixed_eval(model, test_loaders):

    model.eval()
    total_labels = []
    total_pred = []

    for test_loader in test_loaders:
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            total_labels.append(labels)
            total_pred.append(predicted)

    total_pred = torch.cat(total_pred, dim = 0)
    total_labels = torch.cat(total_labels, dim = 0)
    accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
    cm = confusion_matrix(total_pred, total_labels, len(np.unique(total_labels.numpy())))

    return accuracy, cm