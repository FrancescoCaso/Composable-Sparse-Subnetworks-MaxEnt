# some part of code taken from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/lottery

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import *
from train_utils import *
import yaml
import copy
from torch.nn import CrossEntropyLoss
from log import *
from itertools import combinations
from LTH_utils import *
import torch.nn.utils.prune as prune
from models_db import ModelsDB

def convert_to_anomaly_det_targets(target, class_probs, num_classes):
    """
    Convert hard labels to anomaly detection target distributions based on custom probabilities.
    """
    batch_size = target.size(0)
    
    target_probs = torch.zeros(batch_size, num_classes, device=target.device) / num_classes

    flag = 0
    
    for cls, prob in class_probs.items():
        mask = (target == cls)
        if prob < 1.0:
            if flag == 0:
                probs_iter = iter(class_probs.keys())
                prova = next(probs_iter)
                if class_probs[prova] < 1: lucky_one = prova 
                else: lucky_one = next(probs_iter)
                flag = 1
            target_probs[mask] = torch.zeros(num_classes, device=target.device)
            target_probs[mask, lucky_one] += 1.0
        else:  
            target_probs[mask] = torch.zeros(num_classes, device=target.device)
            target_probs[mask, cls] = prob         
            remaining_prob = 0
            target_probs[mask] += remaining_prob
            target_probs[mask, cls] -= remaining_prob
    return target_probs

def convert_to_soft_targets(target, class_probs, num_classes):
    """
    Convert hard labels to soft target distributions based on custom probabilities.
    """
    batch_size = target.size(0)
    
    target_probs = torch.zeros(batch_size, num_classes, device=target.device) / num_classes
    for cls, prob in class_probs.items():
        mask = (target == cls) 
        # target_probs[mask] = torch.zeros(num_classes)  
        if prob==1:
            target_probs[mask, cls] = prob 
        else:
            target_probs[mask] = torch.full((num_classes,), prob, device=target.device) 
        
        # remaining_prob = (1 - prob) / (num_classes - 1)  
        # target_probs[mask] += remaining_prob
        # target_probs[mask, cls] -= remaining_prob
        # exit()
    return target_probs

def generate_class_masks(n_classes:int, k_classes:int=1):
    """
    Generates a list of weight masks for a given number of classes, each mask being a list of length `n_classes`.
    The first mask contains only `1`s (fully active), and subsequent masks contain exactly k `1`s and the rest equal to
    1/n_classes.

    Parameters:
    - n_classes (int): The number of classes, determining the length of each mask.
    - k_classes (int): The number of classes to be set to `1` in each mask. Default is `1`.

    Returns:
    List[List[float]]: A list of weight masks
    """
    result = [[1] * n_classes]
    ones = combinations(range(n_classes), k_classes)
    for ii in ones:
        mask = np.ones(n_classes) / n_classes
        mask[list(ii)] = 1
        result.append(mask.tolist())
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Train an MLP on MNIST with specified classes.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed.')
    return parser.parse_args()

def train(
  model,
  config,
  epochs,
  class_probs, 
  train_loader,
  criterion,
  optimizer,
  logger=None,
  val_loader=None,
  test_loader=None,
  device='cpu',
  rewarded_classes=None,
  max_iter=None,  
  anomaly_flag=False,    
):
    # alpha = 1
    # beta = 0
    # gamma = 0

    if max_iter:
        max_epochs = max_iter
    else:
        max_epochs = epochs

    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    patience = 3

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        total_labels = []
        total_pred = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            y = labels.clone()

            # rewarded_mask = torch.tensor([label.item() in rewarded_classes for label in labels], dtype=torch.bool, device=labels.device)

            if anomaly_flag:
                labels = convert_to_anomaly_det_targets(labels, class_probs, len(config["classes"]))
            else:
                labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]))
            
            # loss = criterion(outputs, labels)
            loss = criterion(F.log_softmax(outputs, dim=1), labels)

            # probs = outputs
            # log_probs = outputs.log()
            # # loss = criterion(F.log_softmax(outputs, dim=1), labels).sum(dim=1)
            # loss = criterion(log_probs, labels).sum(dim=1)
            # # loss = criterion(outputs, labels).sum(dim=1)
            # loss[~rewarded_mask] *= alpha

            # # Penalizza output con bassa entropia sui non-rewarded
            # entropy = -torch.sum(probs * log_probs, dim=1)  # entropy H(p)
            # max_entropy = torch.log(torch.tensor(outputs.size(1), device=outputs.device))  # log(N classi)
            # entropy_loss = max_entropy - entropy  # penalizza se l'entropia è bassa

            # # Applichiamo solo ai non-rewarded
            # entropy_loss[rewarded_mask] = 0.0

            # # Trova max prob per ciascun sample
            # max_probs, _ = probs.max(dim=1)  # shape: (batch_size,)
            # # Definisci margine target: idealmente 1 / num_classes
            # target_margin = 1.0 / outputs.size(1)
            # # Penalizza solo i non-rewarded: quanto max_prob supera il margine
            # penalties = (max_probs - target_margin).clamp(min=0.0)
            # # Applica penalizzazione solo sui non-rewarded
            # penalties[rewarded_mask] = 0.0
            # # Loss finale: media delle penalità > gamma è un peso che scegli
            # margin_loss = penalties.mean()

            # loss = loss.mean() + beta * entropy_loss.mean() + gamma * margin_loss


            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_labels.append(y)
            total_pred.append(predicted)
        total_pred = torch.cat(total_pred, dim=0)
        total_labels = torch.cat(total_labels, dim=0)
        accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
        if logger:
            logger({'epoch':epoch, 'train_accuracy':accuracy})
    # torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             }, logger.results_directory + logger.name + '.pt')

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            total_labels = []
            total_pred = []
            
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                y = labels.clone()
                if anomaly_flag:
                    labels = convert_to_anomaly_det_targets(labels, class_probs, len(config["classes"]))
                else:
                    labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]))
                loss = criterion(F.log_softmax(outputs, dim=1), labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_labels.append(y)
                total_pred.append(predicted)
                
            total_pred = torch.cat(total_pred, dim=0)
            total_labels = torch.cat(total_labels, dim=0)
            accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
            logger({'epoch':epoch, 'val_loss':val_loss, 'val_accuracy':accuracy})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                patience = 3
            else:
                patience -= 1
                
            if patience == 0:
                print(f"Early stopping at epoch {epoch}")
                break

    # Use the best model for final evaluation
    model = best_model
    accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model, test_loader, rewarded_classes=rewarded_classes, device=device)
    return accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy

def main_lth(config, device, args, rw=None, db=None):
    if isinstance(rw, int) and rw is not None:
        rw = [rw]
    if db is None:
        db = ModelsDB()
    
    # config["seeds"] = np.arange(config["seeds"])
    config["seed"] = args.seed

    # anomaly_flag = (config["loss_name"] == "one_vs_all")
    model_name = config["model"]["name"]

    if model_name == 'MLP':
        params = {
            "input_dim": config["model"]["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "output_dim": len(config["classes"]),
            "dropout": config["model"]["dropout"],
            "t": config["model"]["t"],
            "activation_fn": getattr(nn, config["model"]["activation_fn"])
        }
    elif model_name == 'ConvNet':
        params = {
            "input_channels": config["model"]["input_channels"],
            "input_dim": config["model"]["input_dim"],
            "hidden_conv_dims": config["model"]["hidden_conv_dims"],
            "hidden_fc_dims": config["model"]["hidden_fc_dims"],
            "output_dim": len(config["classes"]),
            "dropout_rate": config["model"]["dropout_rate"],
            "activation_fn": getattr(nn, config["model"]["activation_fn"]),
            "conv_kernel_size": config["model"]["conv_kernel_size"], 
            "padding": config["model"]["padding"],
            "conv_stride": config["model"]["conv_stride"],
            "pool_kernel_size": config["model"]["pool_kernel_size"],
            "pool_stride": config["model"]["pool_stride"],
            "t": config["model"]["t"],
        }

    config['classes'] = ClassesList(config["classes"], config["dataset_name"])
    train_loader, val_loader, test_loader = get_loaders(config['classes'], config["batch_size"])

    if rw is None:
        combs = generate_class_masks(len(config["classes"]), args.n_classes)
    else:
        combs = [[1 if i in rw else 1/len(config["classes"]) for i in range(len(config["classes"]))]]

    for ind, comb in enumerate(combs):
        rewarded_classes = [i for i, w in enumerate(comb) if w == 1]

        if db.query(config, config["seed"], rewarded_classes):
            print(f"Already trained with this configuration and rewarded class {rewarded_classes}")
            out = None
            continue

        epochs = config["epochs"]
        config["weights"] = comb
        
        logger = initialize_logger_from_config(config)
        logger.log(config)
        
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = create_model(config['model']['name'], params)
        model_copy = create_model(config['model']['name'], params)
        model_copy.load_state_dict(model.state_dict())

        model.to(device)
        model_copy.to(device)

        class_probs = {np.arange(len(config["classes"]))[i]:config["weights"][i] for i in range(len(config["classes"]))}
        # criterion = CrossEntropyLoss()
        criterion = nn.KLDivLoss(reduction="batchmean")
        # criterion = nn.KLDivLoss(reduction="none")
        optimizer = get_optimizer(model.parameters(), **config['optimizer'])
        
        # Train and prune loop
        prune_ratio = config["prune_ratio"]
        prune_iter = config["prune_iter"]
        max_iter = epochs
        if prune_ratio > 0:
            per_round_prune_ratio = 1 - (1 - prune_ratio) ** (
                1 / prune_iter
            )
            prunable_layers = [
                module for module in model.modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]
            per_round_prune_ratios = [per_round_prune_ratio] * len(prunable_layers)
            per_round_prune_ratios[-1] /= 2

            per_round_max_iter = int(max_iter / prune_iter)

            for prune_it in range(prune_iter):
                accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = train(
                    model,
                    config,
                    epochs,
                    class_probs,
                    train_loader,
                    criterion,
                    optimizer,
                    logger,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    rewarded_classes=rewarded_classes,
                    max_iter=per_round_max_iter,
                )
                logger({'test_accuracy':accuracy, 'rewarded_accuracy':rewarded_accuracy, 'mean_non_rewarded_entropy':mean_non_rewarded_entropy})
                prune_model(model, per_round_prune_ratios)

                copy_weights_model(model_copy, model)

                # stats = compute_stats(model)
                # for name, stat in stats.items():
                #     summary_name = f"{name}_pruneiter={prune_it}"
                #     wandb.run.summary[summary_name] = stat
        # logger({'test_accuracy':accuracy, 'cost_matrix':cm})
        # if args.reinitialize == "true":
        #     reinit_mlp(model)

        # Run actual training with a final pruned network

        print(f"Pruned model performance")

        accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = train(
                    model,
                    config,
                    epochs,
                    class_probs,
                    train_loader,
                    criterion,
                    optimizer,
                    logger=logger,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    rewarded_classes=rewarded_classes,
                )
        logger({'test_accuracy':accuracy, 'cost_matrix':cm, 'rewarded_accuracy':rewarded_accuracy, 'mean_non_rewarded_entropy':mean_non_rewarded_entropy})
        config['model_path'] = (logger.results_directory / f'{logger.name}.pt').as_posix()
        # save_pth = f"check_mlp_{'_'.join([str(k) for k in config['model']['hidden_dims']])}_lth_seed_{args.seed}_prune_ratio_{prune_ratio}_comb_{''.join(map(str, comb))}_{dataset_name}.pth"
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, config['model_path'])
        logger.finish()
        out = config, logger.name
    return out
        # torch.save({
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             }, save_pth)
        # print(f"Modello salvato come {save_pth}")

def main_tetris(config, device, args):

    config["seed"] = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["classes"].sort()

    # dataset_name = config["dataset_name"]
    # anomaly_flag = (config["loss_name"] == "one_vs_all")

    model_name = config["model"]["name"]

    if model_name == 'MLP':
        params = {
            "input_dim": config["model"]["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "output_dim": len(config["classes"]),
            "dropout": config["model"]["dropout"],
            "t": config["model"]["t"],
            "activation_fn": getattr(nn, config["model"]["activation_fn"])
        }
    elif model_name == 'ConvNet':
        params = {
            "input_channels": config["model"]["input_channels"],
            "input_dim": config["model"]["input_dim"],
            "hidden_conv_dims": config["model"]["hidden_conv_dims"],
            "hidden_fc_dims": config["model"]["hidden_fc_dims"],
            "output_dim": len(config["classes"]),
            "dropout_rate": config["model"]["dropout_rate"],
            "activation_fn": getattr(nn, config["model"]["activation_fn"])
        }

    if config["dataset_name"] == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_loaders(config["classes"], config["batch_size"])
    elif config["dataset_name"] == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_loaders(config["classes"], config["batch_size"])
    elif config["dataset_name"] == 'fashion_mnist':
        train_loader, val_loader, test_loader = get_fashionmnist_loaders(config["classes"], config["batch_size"])

    combs = generate_class_masks(len(config["classes"]), args.n_classes)

    # l = len(combs) - 1
    masks = {}

    for i, comb in enumerate(combs):
        config["weights"] = comb
        logger = initialize_logger_from_config(config)
        logger.log(config)
        
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = create_model(config['model']['name'], params)
        model_copy = create_model(config['model']['name'], params)
        model_copy.load_state_dict(model.state_dict())

        model.to(device)
        model_copy.to(device)

        if i > 1:
            for name, module in model.named_modules():
                for param in ['weight', 'bias']:  # A seconda di cosa hai prunato
                    mask_key = f"{name}.{param}_mask"
                    if mask_key in masks:
                        # Applica il pruning custom con la maschera
                        prune.custom_from_mask(module, name=param, mask=torch.ones_like(masks[mask_key])-masks[mask_key])

        class_probs = {np.arange(len(config["classes"]))[i]:config["weights"][i] for i in range(len(config["classes"]))}
        criterion = CrossEntropyLoss()
        optimizer = get_optimizer(model.parameters(), **config['optimizer'])
        
        # Train and prune loop
        prune_ratio = config["prune_ratio"]
        prune_iter = config["prune_iter"]
        max_iter = config["epochs"]
        if prune_ratio > 0:
            per_round_prune_ratio = 1 - (1 - prune_ratio) ** (
                1 / prune_iter
            )

            per_round_prune_ratios = [per_round_prune_ratio] * len(model.layers)
            per_round_prune_ratios[-1] /= 2

            per_round_max_iter = int(max_iter / prune_iter)

            for prune_it in range(prune_iter):
                accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = train(
                    model,
                    config,
                    class_probs,
                    train_loader,
                    criterion,
                    optimizer,
                    logger,
                    test_loader,
                    device,
                    per_round_max_iter,
                )
                prune_mlp(model, per_round_prune_ratios)

                copy_weights_mlp(model_copy, model)

                # stats = compute_stats(model)
                # for name, stat in stats.items():
                #     summary_name = f"{name}_pruneiter={prune_it}"
                #     wandb.run.summary[summary_name] = stat

        # if args.reinitialize == "true":
        #     reinit_mlp(model)

        # Run actual training with a final pruned network

        print(f"Pruned model performance")

        accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = train(
                    model,
                    config,
                    class_probs,
                    train_loader,
                    criterion,
                    optimizer,
                    logger,
                    test_loader,
                    device,
                )

        save_pth = f"tetris_mlp_{'_'.join([str(k) for k in config['model']['hidden_dims']])}_lth_seed_{args.seed}_prune_ratio_{prune_ratio}_comb_{''.join(map(str, comb))}_{config['dataset_name']}.pth"
        torch.save(model.state_dict(), save_pth)
        print(f"Modello salvato come {save_pth}")

        logger({'test_accuracy':accuracy, 'cost_matrix':cm, 'rewarded_accuracy':rewarded_accuracy, 'mean_non_rewarded_entropy':mean_non_rewarded_entropy})

        if i == 1:
            for name, module in model.named_modules():
                for attr_name in dir(module):
                    if attr_name.endswith('_mask'):
                        mask = getattr(module, attr_name)
                        masks[f"{name}.{attr_name}"] = mask.clone()
        elif i > 1:
            for name, module in model.named_modules():
                for attr_name in dir(module):
                    if attr_name.endswith('_mask'):
                        mask = getattr(module, attr_name)
                        masks[f"{name}.{attr_name}"] = torch.maximum(mask.clone(), masks[f"{name}.{attr_name}"].clone())
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    main_lth(config, device, args)
    # main_tetris(config, device, args)