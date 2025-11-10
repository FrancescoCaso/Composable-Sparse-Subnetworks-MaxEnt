import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from train_utils import get_loaders, eval, ClassesList

import yaml
import copy
from log import *
import typer
from typing import List  
from itertools import combinations
from dataset_creation import get_model_config, retrieve_model, retrieve_model_old
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models_db import ModelsDB

db = ModelsDB()


def add_model_weights(model1, model2, weight_names=None):
    """
    Adds the weights of model2 to model1 for the specified weight names.
    
    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model to add to the first model.
        weight_names (list of str, optional): List of parameter names to aggregate.
                                              If None, all weights will be added.
                                              
    Returns:
        nn.Module: A new model with aggregated weights.
    """
    new_model = copy.deepcopy(model1)
    # new_model = type(model1)()  # Crea una nuova istanza della stessa classe
    # new_model.load_state_dict(model1.state_dict())

    
    state_dict1 = new_model.state_dict()
    state_dict2 = model2.state_dict()
    
    available_names = list(state_dict1.keys())
    
    if weight_names is None:
        weight_names = available_names 
    
    missing_names = [name for name in weight_names if name not in available_names]
    
    if missing_names:
        raise ValueError(f"Invalid weight names: {missing_names}. "
                         f"Available names are: {available_names}")
    
    with torch.no_grad():
        for name in weight_names:
            state_dict1[name].add_(state_dict2[name])
    
    new_model.load_state_dict(state_dict1)
    
    return new_model

def add_model_weights_lth(model1, model2, weight_names=None):
    """
    Adds the weights of model2 to model1 for the specified weight names.
    
    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model to add to the first model.
        weight_names (list of str, optional): List of parameter names to aggregate.
                                              If None, all weights will be added.
                                              
    Returns:
        nn.Module: A new model with aggregated weights.
    """
    # Create a new model instance instead of deep copying
    # new_model = create_model(model1.name, model1.config)
    # new_model = type(model1)()  # Crea una nuova istanza della stessa classe
    # new_model.load_state_dict(model1.state_dict())
    new_model = copy.deepcopy(model1)

    
    state_dict1 = model1.state_dict().copy()
    state_dict2 = model2.state_dict().copy()
    state_dict = state_dict1.copy()

    for a, b in state_dict1.items():
        if "orig" in a:
            state_dict[a] = state_dict[a] * state_dict[a[:-4] +'mask']

    for a, b in state_dict2.items():
        if "orig" in a:
            state_dict[a] += state_dict2[a] * state_dict2[a[:-4] +'mask']
        if "mask" in a:
            state_dict[a] = torch.maximum(state_dict[a], state_dict2[a])

    # eps = 1e-6

    # model = MLP(input_dim=784, hidden_dims=[1000], output_dim=10)

    # for name, module in model.named_modules():
    #     if hasattr(module, 'weight'):
    #         prune.custom_from_mask(module, name='weight', mask=torch.ones_like(module.weight))
    #     if hasattr(module, 'bias') and module.bias is not None:
    #         prune.custom_from_mask(module, name='bias', mask=torch.ones_like(module.bias))

    new_model.load_state_dict(state_dict)
    
    return new_model

def plot_confusion_matrix(cm:np.array, 
                          class_names: list[str] = None, 
                          title:str = "Confusion Matrix", 
                          output: str = None,
                          ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, square=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real")
    ax.set_title(title)
    if output:
        print('Saving image to', output)
        plt.tight_layout()
        plt.savefig(output)
    else:
        print(pd.DataFrame(cm, columns=class_names, index=class_names))

app = typer.Typer()


@app.command()
def incremental_comparison_statistics(config: str = typer.Argument(..., help='Path to the experiment configuration file'),
                           classes: List[int] = typer.Option(..., "-c", help="List of rewarded classes"), 
                           layers: List[str] = typer.Option(None, "-l", help="List of layers to add"), 
                           aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
                           output: str = typer.Option(None, "-o", "--output", help="Path to the output file."),
                           seeds: List[int] = typer.Option(..., "-s", help="List of seeds to use for the models."),
                           is_val: bool = typer.Option(False, "--is-val", help='Use validation set.'),                       
                           calc_loss_barrier: bool = typer.Option(False, "-lb", "--lossbarrier", help="Enable loss barrier computation."), 
                           chosen_loader: str = typer.Option('train', "-l", "--loader", help="Train, val, or test."),                        
                           relative: bool = typer.Option(True, "-r", "--relative", help="Absolute or relative barrier values."),                        
                           num_alphas: int = typer.Option(11, "-a", "--alphas", help="How many alphas to use in the barrier."),                        
                           reduction: str = typer.Option("sum", "-re", "--reduction", help="Whether use sum or mean as KLDivLoss reduction")
                                                    ):
    """
    Performs incremental comparison across multiple seeds and computes statistics.
    For each step in the incremental process, computes mean and standard deviation
    of accuracy, rewarded accuracy, and entropy across all seeds.
    """
    from loss_barrier import loss_barrier
    
    if output:
        os.makedirs(output, exist_ok=True)
    
    if not isinstance(config, dict):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

    all_cms = [[] for _ in range(len(classes))]

    # Run for each seed
    res_df = pd.DataFrame(columns=['seed', 'accuracy', 'rewarded_accuracy', 'entropy', 'level'])
    for seed in seeds:
        print(f"\nProcessing seed {seed}")
        config["classes"] = ClassesList(config["classes"], config["dataset_name"])
        this_config = get_model_config(config, seed, classes[0], db=db)
        model, name = retrieve_model(this_config, aligned, device=device)
        
        _, val_loader, test_loader = get_loaders(config['classes'], 512)

        final_name = name
        for ind, rewarded_class in enumerate(classes):
            if ind > 0:
                this_config = get_model_config(config, seed, rewarded_class, db=db)
                model2, name2 = retrieve_model(this_config, aligned, device=device)
                
                if calc_loss_barrier:
                    loss_barrier(
                        model_1=model, model_2=model2, config=config,
                        classes = classes[:ind+1],
                        aligned=aligned,
                        chosen_loader=chosen_loader,
                        output=output + "lb_" + final_name  + f'+{name2}' + '.png',
                        relative=relative,
                        num_alphas=num_alphas,
                        reduction=reduction,
                        seed=seed
                          )
                    
                if this_config["lth"]:
                    model = add_model_weights_lth(model, model2)
                else:
                    model = add_model_weights(model, model2)
                final_name += f'+{name2}'

            if is_val:
                acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model, val_loader, rewarded_classes=classes[:ind+1], device=device)
            else:
                acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model, test_loader, rewarded_classes=classes[:ind+1], device=device)

            res_df = pd.concat([res_df, pd.DataFrame([[seed, acc, rewarded_accuracy, mean_non_rewarded_entropy, ind+1]], columns=res_df.columns)], ignore_index=True)
            all_cms[ind].append(cm)

    # Compute statistics and print results
    print("\nStatistics across all seeds:")
    tabprint = pd.DataFrame(columns=["Step", "Classes", "Mean Acc ± Std", "Mean Rewarded Acc ± Std", "Mean Entropy ± Std"])

    for ind in res_df.level.unique():
        mean_acc = res_df[res_df.level == ind].accuracy.mean()
        std_acc = res_df[res_df.level == ind].accuracy.std()
        mean_rew_acc = res_df[res_df.level == ind].rewarded_accuracy.mean()
        std_rew_acc = res_df[res_df.level == ind].rewarded_accuracy.std()
        mean_ent = res_df[res_df.level == ind].entropy.mean()
        std_ent = res_df[res_df.level == ind].entropy.std()

        tabprint = pd.concat([tabprint, pd.DataFrame([[ind, classes[:ind], f"{mean_acc:.3f} ± {std_acc:.3f}", f"{mean_rew_acc:.3f} ± {std_rew_acc:.3f}", f"{mean_ent:.3f} ± {std_ent:.3f}"]], columns=tabprint.columns)], ignore_index=True)
    print(tabprint)

    # Plot average confusion matrix for each step
    if output:
        for ind in range(len(classes)):
            avg_cm = np.mean(all_cms[ind], axis=0)
            title = f"Average CM (Step {ind+1}, Classes {classes[:ind+1]})"
            if layers:
                title += f"_{layers}"
            if aligned:
                title += "_aligned"
            
            output_path = os.path.join(output, f"step_{ind+1}_avg_cm.pdf")
            plot_confusion_matrix(avg_cm, class_names=config['classes'], title=title, output=output_path)
    return res_df

@app.command()
def single_comparison_statistics(
    config: str = typer.Argument(..., help='Path to the experiment configuration file'),
    class1: List[int] = typer.Option(..., "--class1", "-c1", help='First list of rewarded classes'),
    class2: List[int] = typer.Option(..., "--class2", "-c2", help='Second list of rewarded classes'),
    layers: List[str] = typer.Option(None, "-l", "--layers", help="List of layers to add"), 
    aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
    output: str = typer.Option(None, "-o", "--output", help="Path to the output file."),
    seeds: List[int] = typer.Option(..., "-s", "--seeds", help="List of seeds to use for the models."),
    is_val: bool = typer.Option(False, "--is-val", help='Use validation set.'),                    
    calc_loss_barrier: bool = typer.Option(False, "-lb", "--lossbarrier", help="Enable loss barrier computation."), 
    chosen_loader: str = typer.Option('train', "-l", "--loader", help="Train, val, or test."),                        
    relative: bool = typer.Option(True, "-r", "--relative", help="Absolute or relative barrier values."),                        
    num_alphas: int = typer.Option(11, "-a", "--alphas", help="How many alphas to use in the barrier."),                        
    reduction: str = typer.Option("sum", "-re", "--reduction", help="Whether use sum or mean as KLDivLoss reduction"),
    used: bool = False,
):
    """
    Performs single comparison between two models across multiple seeds and computes statistics.
    For each seed, merges two models (one trained for class1 list and one for class2 list) and computes metrics.
    Then calculates mean and standard deviation of accuracy, rewarded accuracy, and entropy across all seeds.
    
    Args:
        config (str): Path to the experiment configuration file
        class1 (List[int]): First list of rewarded classes
        class2 (List[int]): Second list of rewarded classes
        layers (List[str], optional): List of layers to add
        aligned (bool, optional): Use the aligned model
        output (str, optional): Path to the output file
        seeds (List[int]): List of seeds to use for the models
        is_val (bool, optional): Use validation set
    """
    from loss_barrier import loss_barrier
    if output:
        os.makedirs(output, exist_ok=True)

    if not isinstance(config, dict):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

    # Initialize lists to store metrics
    all_accuracies = []
    all_rewarded_accuracies = []
    all_entropies = []
    all_cms = []

    # Run for each seed
    for seed in seeds:
        print(f"\nProcessing seed {seed}")
        
        config["classes"] = ClassesList(config["classes"], config["dataset_name"])
        config1 = get_model_config(config, seed, class1, db=db)
        config2 = get_model_config(config, seed, class2, db=db)
        if config1 is None or config2 is None:
            print(f"Skipping seed {seed} due to missing configuration.")
            seeds.remove(seed)
            continue
            #TODO: remove the current seed from the list
        model_1, name1 = retrieve_model(config1, aligned, device=device)
        model_2, name2 = retrieve_model(config2, aligned, device=device)
        
        # we are assuming both models come from the same dataset
        _, val_loader, test_loader = get_loaders(config['classes'], 512)  

        _, _, _, _ = eval(model_1, val_loader, rewarded_classes=class1, device=device)

        if config["lth"]:
            merged_model = add_model_weights_lth(model_1, model_2, weight_names=layers)
        else:
            merged_model = add_model_weights(model_1, model_2, weight_names=layers)

        # Combine both lists of rewarded classes for evaluation
        all_rewarded_classes = class1 + class2

        if calc_loss_barrier:
            loss_barrier(
                model_1=model_1, model_2=model_2, config=config,
                classes = all_rewarded_classes,
                aligned=aligned,
                chosen_loader=chosen_loader,
                output=output + "lb_" + f"{name1}+{name2}" + '.png',
                relative=relative,
                num_alphas=num_alphas,
                reduction=reduction,
                seed=seed
                  )

        if is_val:
            acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(merged_model, val_loader, rewarded_classes=all_rewarded_classes, device=device)
        else:
            acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(merged_model, test_loader, rewarded_classes=all_rewarded_classes, device=device)
        
        # Store metrics
        all_accuracies.append(acc)
        all_rewarded_accuracies.append(rewarded_accuracy)
        all_entropies.append(mean_non_rewarded_entropy)
        all_cms.append(cm)

    # Compute statistics and print results
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_rew_acc = np.mean(all_rewarded_accuracies)
    std_rew_acc = np.std(all_rewarded_accuracies)
    mean_ent = np.mean(all_entropies)
    std_ent = np.std(all_entropies)

    print("\nStatistics across all seeds:")
    print(f"Classes: {class1} + {class2}")
    print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean Rewarded Accuracy: {mean_rew_acc:.3f} ± {std_rew_acc:.3f}")
    print(f"Mean Entropy: {mean_ent:.3f} ± {std_ent:.3f}")

    # Plot average confusion matrix
    if output:
        avg_cm = np.mean(all_cms, axis=0)
        title = f"{name1}+{name2}"
        if layers:
            title += f"_{layers}"
        if aligned:
            title += "_aligned"
        
        output_path = os.path.join(output, f"{title}_avg_cm.pdf")
        plot_confusion_matrix(avg_cm, class_names=config['classes'], title=title, output=output_path)
    
    if used:
        return mean_acc, std_acc, mean_rew_acc, std_rew_acc, mean_ent, std_ent

    return pd.DataFrame({
        "seed": seeds,
        "accuracy": all_accuracies,
        "rewarded_accuracy": all_rewarded_accuracies,
        "entropy": all_entropies,
    })
    

@app.command()
def multiple_comparison_statistics(config: str = typer.Argument(..., help='Path to the experiment configuration file'),
                        classes: List[int] = typer.Option(None, "-c", help="List of rewarded classes"), 
                        layers: List[str] = typer.Option(None, "-l", help="List of layers to add"), 
                        aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
                        no_show: bool = typer.Option(False, "--no_show", help='If you want to show the figures or not'),
                        output: str = typer.Option(None, "-o", "--output", help="Path to the output file"),
                        seeds: List[int] = typer.Option(..., "-s", help="List of seeds to use for the models."),
                        is_val: bool = typer.Option(False, "--is-val", help='Use validation set.'),                       
                        calc_loss_barrier: bool = typer.Option(False, "-lb", "--lossbarrier", help="Enable loss barrier computation."), 
                        chosen_loader: str = typer.Option('train', "-l", "--loader", help="Train, val, or test."),                        
                        relative: bool = typer.Option(True, "-r", "--relative", help="Absolute or relative barrier values."),                        
                        num_alphas: int = typer.Option(11, "-a", "--alphas", help="How many alphas to use in the barrier."),                        
                        reduction: str = typer.Option("sum", "-re", "--reduction", help="Whether use sum or mean as KLDivLoss reduction")
                        ):
    from loss_barrier import loss_barrier
    os.makedirs(output, exist_ok=True)
    print(list(combinations(classes, 2)))
    prints = []
    for class1,class2 in list(combinations(classes, 2)):
        if class1 == class2:
            continue
        mean_acc, std_acc, mean_rew_acc, std_rew_acc, mean_ent, std_ent = single_comparison_statistics(config, [class1], [class2], layers, aligned, output, seeds, is_val,
                                                                                                      calc_loss_barrier, chosen_loader, relative, num_alphas, reduction, used=True)
        prints.append([class1, class2, mean_acc, std_acc, mean_rew_acc, std_rew_acc, mean_ent, std_ent])
    with open("exp3_output.txt", "w") as f:
        for class1, class2, mean_acc, std_acc, mean_rew_acc, std_rew_acc, mean_ent, std_ent in prints:
            f.write(f"Classes: {class1} + {class2}\n")
            f.write(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}\n")
            f.write(f"Mean Rewarded Accuracy: {mean_rew_acc:.3f} ± {std_rew_acc:.3f}\n")
            f.write(f"Mean Entropy: {mean_ent:.3f} ± {std_ent:.3f}\n")
            f.write("\n")  # Per separare i blocchi


@app.command()
def single_evaluation_statistics(config: str = typer.Argument(..., help='Path to the experiment configuration file'),
                      class1: int = typer.Argument(None, help='First class rewarded'),
                      aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
                      output: str = typer.Option(None, "-o", "--output", help="Path to the output file."),
                      seeds: List[int] = typer.Option(..., "-s", help="List of seeds to use for the models."),
                      is_val: bool = typer.Option(False, "--is-val", help='Use validation set.'),
                                                    ):
    """
    Performs single evaluation across multiple seeds and computes statistics.
    For each seed, evaluates a model trained for the specified class and computes metrics.
    Then calculates mean and standard deviation of accuracy, rewarded accuracy, and entropy across all seeds.
    """
    if output:
        os.makedirs(output, exist_ok=True)
    
    if not isinstance(config, dict):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

    # Initialize lists to store metrics
    all_accuracies = []
    all_rewarded_accuracies = []
    all_entropies = []
    all_cms = []

    # Run for each seed
    for seed in seeds:
        print(f"\nProcessing seed {seed}")
        
        config["classes"] = ClassesList(config["classes"], config["dataset_name"])
        this_config = get_model_config(config, seed, class1, db=db)
        model_1, name1 = retrieve_model(this_config, aligned, device=device)

        _, val_loader, test_loader = get_loaders(this_config['classes'], this_config["batch_size"])

        if is_val:
            acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model_1, val_loader, rewarded_classes=[class1], device=device)
        else:
            acc, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model_1, test_loader, rewarded_classes=[class1], device=device)
        
        # Store metrics
        all_accuracies.append(acc)
        all_rewarded_accuracies.append(rewarded_accuracy)
        all_entropies.append(mean_non_rewarded_entropy)
        all_cms.append(cm)

    # Compute statistics and print results
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_rew_acc = np.mean(all_rewarded_accuracies)
    std_rew_acc = np.std(all_rewarded_accuracies)
    mean_ent = np.mean(all_entropies)
    std_ent = np.std(all_entropies)


    # Plot average confusion matrix
    if output:
        print("\nStatistics across all seeds:")
        print(f"Class: {class1}")
        print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"Mean Rewarded Accuracy: {mean_rew_acc:.3f} ± {std_rew_acc:.3f}")
        print(f"Mean Entropy: {mean_ent:.3f} ± {std_ent:.3f}")
        avg_cm = np.mean(all_cms, axis=0)
        title = f"{name1}"
        if aligned:
            title += "_aligned"
        
        output_path = os.path.join(output, f"{title}_avg_cm.pdf")
        plot_confusion_matrix(avg_cm, class_names=config['classes'], title=title, output=output_path)
    
    return pd.DataFrame({
        "seed": seeds,
        "accuracy": all_accuracies,
        "rewarded_accuracy": all_rewarded_accuracies,
        "entropy": all_entropies,
    })

def main():
    app()


if __name__ == "__main__":
    app()