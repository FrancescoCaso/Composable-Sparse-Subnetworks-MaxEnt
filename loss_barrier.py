import os
import numpy as np
import torch
from train_utils import get_loaders, eval, ClassesList
import torch.nn as nn
# from lobotomy_2 import *
import yaml
import copy
from log import *
import typer
from typing import List
from pathlib import PosixPath, Path
from frankenstein_sample_level import retrieve_model, retrieve_model, get_model_config, device
from torch.functional import F
import matplotlib.pyplot as plt
import pickle


from torchmetrics.classification import MulticlassCalibrationError



app = typer.Typer()

def convert_to_soft_targets(target, class_probs, num_classes, config=None):
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


def plot_front(front, alphas, barrier, sum_merge=None, filename=None, relative=None, losses_m1_varying=None, losses_m2_varying=None):
    """
    Plots the Pareto front, considering alphas and a barrier value.

    Args:
        front: A list of loss values representing the Pareto front.
        alphas: A list of alpha values corresponding to the Pareto front points.
        barrier: The barrier value to mark on the plot.
        sum_merge: An optional value to plot as "sum merge".
        filename: An optional filename to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, front, marker='o', linestyle='-')
    plt.axhline(y=barrier, color='r', linestyle='--', label='Barrier')
    plt.scatter(alphas[front.index(max(front))], max(front), color='red', marker='x', s=100, label='Barrier position')


    if sum_merge is not None:
        plt.axhline(y=sum_merge, color='g', linestyle='--', label='Sum Merge')
    if losses_m1_varying is not None:
        plt.plot(alphas, losses_m1_varying, color='g', marker='>', linestyle='-', label='Sum Merge varying M1')
    if losses_m2_varying is not None:
        plt.plot(alphas, losses_m2_varying, color='g', marker='<', linestyle='-', label='Sum Merge varying M2')


    plt.xlabel('Alpha')
    if relative is not None and relative == True:
        plt.ylabel("Loss %")
    else:
        plt.ylabel('Loss')
    plt.title('Loss Barrier Front')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)

    plt.show()




import numpy as np
import matplotlib.pyplot as plt

def plot_front_new(front,
               alphas,
               barrier,
               sum_merge=None,
               filename=None,
               relative=False,
               losses_m1_varying=None,
               losses_m2_varying=None,
               eces=None,
               eces_m1_varying=None,
               eces_m2_varying=None):
    """
    • Builds new_alphas = linspace(0,1,2N-1)
    • Plots `front` only at the original alphas (N points)
    • Builds new_losses = losses_m2_varying + reversed(losses_m1_varying)[1:]
      so it's length 2N-1 and aligns with new_alphas.
    • Adds two extra bottom x-axes: 
      – M2 shows [α0…αN-1, 1…1] 
      – M1 shows [1…1, αN-1…α0]
    """
    N = len(alphas)
    # 1) make the extended α‐axis
    new_alphas = np.linspace(0, 1, num=2*N - 1, endpoint=True)

    # 2) build the concatenated "varying" losses list
    #    note: losses_m1_varying and losses_m2_varying are Python lists
    new_losses = (
        list(losses_m2_varying) +
        list(losses_m1_varying[::-1])[1:]
    )

    new_eces = (
        list(eces_m2_varying) +
        list(eces_m1_varying[::-1])[1:]
    )

    # 3) plot
    fig, ax = plt.subplots(figsize=(8,6))

    # ── main front & barrier & optional sum_merge ─────────────
    ax.plot(alphas, front[::-1],       'o-', label='Interpolation Loss')
    ax.axhline(barrier, color='r', linestyle='--', label='Barrier')
    # scatter only at the max front
    imax = np.argmax(front)
    ax.scatter(alphas[::-1][imax], front[imax],
               color='r', marker='x', s=100, label='Barrier Point')
    ax.plot(alphas, eces[::-1],       '.-', label='ECE Interpolation Loss')

    # ── the combined varying-loss curve ───────────────────────
    ax.plot(new_alphas, new_losses,
            '-^',
            label='Sum Loss')
    ax.plot(new_alphas, new_eces,
            '-2',
            label='ECE Sum Loss')

    ax.set_xlabel('1 - Alpha')
    ax.set_ylabel('Loss %' if relative else 'Loss')
    ax.set_title('Loss Barrier Front')
    ax.grid(True)

    # 4) define the tick‐label sequences for the two extras
    #    M2: [α0…αN-1] + [1,1,…,1] (N-1 ones)
    m2_lbls = list(alphas) + [1.0]*(N-1)
    #    M1: [1,1,…,1] (N-1 ones) + [αN-1…α0]
    m1_lbls = [1.0]*(N-1) + list(alphas[::-1])

    # 5) extra bottom axis for M2
    sec2 = ax.secondary_xaxis(-0.30, functions=(lambda x: x, lambda x: x))
    sec2.set_xlabel('M2', loc="right")
    sec2.set_xticks(new_alphas)
    sec2.set_xticklabels( [f"{v:.2f}" if (v != 0.0 and v != 1.0) else f"{v:.0f}" for v in m2_lbls])

    # 6) extra bottom axis for M1, offset downward a bit
    sec1 = ax.secondary_xaxis(-0.15, functions=(lambda x: x, lambda x: x))
    sec1.set_xlabel('M1', loc="left")
    sec1.set_xticks(new_alphas)
    sec1.set_xticklabels( [f"{v:.2f}" if (v != 0.0 and v != 1.0) else f"{v:.0f}" for v in m1_lbls])

    if sum_merge is not None:
        ax.axhline(sum_merge, color='g', linestyle='--', label='Sum Merge Barrier')
            
    # 7) legend, layout, save/show
    ax.legend(loc='best')
    fig.subplots_adjust(bottom=0.30)  # make room for the two extra axes
    if filename:
        fig.savefig(filename)
    plt.show()


def interpolate_model_weights_lth(model1, model2, wgs_1=1., wgs_2=1.):
    """
    Adds the weights of model2 to model1 for the specified weight names.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model to add to the first model.

    Returns:
        nn.Module: A new model with aggregated weights.
    """

    new_model = copy.deepcopy(model1)

    state_dict1 = model1.state_dict().copy()
    state_dict2 = model2.state_dict().copy()
    state_dict = state_dict1.copy()

    for a, b in state_dict1.items():
        if "orig" in a:
            state_dict[a] = wgs_1 * state_dict[a] * state_dict[a[:-4] +'mask']

    for a, b in state_dict2.items():
            if "orig" in a:
                state_dict[a] += wgs_2 * state_dict2[a] * state_dict2[a[:-4] +'mask']
            if "mask" in a:
                state_dict[a] = torch.maximum(state_dict[a], state_dict2[a])

    new_model.load_state_dict(state_dict)

    return new_model


@app.command()
def loss_barrier_single(config: str = typer.Argument(..., help='Path to the experiment configuration file'),
                        classes: List[int] = typer.Option(..., "-c", help="List of rewarded classes"),
                        aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
                        chosen_loader: str = typer.Option('train', "-l", "--loader", help="Train, val, or test."),                        
                        output: str = typer.Option(None, "-o", "--output", help="Name of the output file."),               
                        seed: int = typer.Option(0, "-s", "--seed", help="Seed to use for the model."),         
                        relative: bool = typer.Option(True, "-r", "--relative", help="Absolute or relative barrier values."),                        
                        num_alphas: int = typer.Option(11, "-a", "--alphas", help="How many alphas to use in the barrier."),                        
                        reduction: str = typer.Option("sum", "-re", "--reduction", help="Whether use sum or mean as KLDivLoss reduction")):
    
    assert len(classes) == 2, "il numero di classi deve essere 2"

    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    config["classes"] = ClassesList(config["classes"], config["dataset_name"])
    model_1, name_1 = retrieve_model(get_model_config(config, seed, classes[0]), aligned, device=device)
    model_2, name_2 = retrieve_model(get_model_config(config, seed, classes[1]), aligned, device=device)

    return loss_barrier(
                model_1=model_1, model_2=model_2, config=config,
                classes = classes,
                aligned=aligned,
                chosen_loader=chosen_loader,
                output=output,
                relative=relative,
                num_alphas=num_alphas,
                reduction=reduction,
                seed=seed
              )




#@app.command()
@torch.no_grad()
def loss_barrier(model_1, model_2, config,
                        #config: str = typer.Argument(..., help='Path to the experiment configuration file'),
                        classes: List[int] = typer.Option(..., "-c", help="List of rewarded classes"),
                        aligned: bool = typer.Option(False, "--aligned", help='Use the aligned model'),
                        chosen_loader: str = typer.Option('train', "-l", "--loader", help="Train, val, or test."),                        
                        output: str = typer.Option(None, "-o", "--output", help="Name of the output file."),                        
                        relative: bool = typer.Option(True, "-r", "--relative", help="Absolute or relative barrier values."),                        
                        num_alphas: int = typer.Option(11, "-a", "--alphas", help="How many alphas to use in the barrier."),                        
                        reduction: str = typer.Option("sum", "-re", "--reduction", help="Whether use sum or mean as KLDivLoss reduction"),
                 seed = 0
                ):

    if output:
        # get the directory part of the path
        out_dir = os.path.dirname(output)
        # only mkdir if there actually is a directory component
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_dir = os.path.dirname("./pickles_lb/")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)        

    #config["classes"].sort()
    loaders =  get_loaders(config['classes'], config["batch_size"])   

    loader = loaders[{'train' : 0, 'val' : 1, 'test' : 2}[chosen_loader]]

    class_probs = {np.arange(len(config["classes"]))[i]: (1. if config["classes"][i] in classes else 1./len(config["classes"])) for i in range(len(config["classes"]))}
    print(class_probs)
    loss_fn = nn.KLDivLoss(reduction="sum")

    alphas = np.linspace(0, 1, num=num_alphas, endpoint=True)

    #with torch.no_grad():
    losses = []
    for model in [model_1, model_2]:
        model.eval()
        running_loss = 0.
        num_instances = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]), config)
            loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
            running_loss += loss.item()
            num_instances += len(images)
        if reduction == "mean":
            running_loss = running_loss / num_instances
        losses.append(running_loss)
    eps_bar = (losses[0] + losses[1])/2
    interp_erorrs = [losses[0]* alpha + losses[1]*(1-alpha) for alpha in np.linspace(0, 1, num=2*len(alphas) - 1, endpoint=True)]
    

    losses = []
    eces = []
    for alpha in alphas:
        if config["lth"]:
            model = interpolate_model_weights_lth(model_1, model_2, wgs_1=alpha, wgs_2=1-alpha)
        else:
            assert False
        running_loss = 0.
        num_instances = 0
        model.eval()
        ece = MulticlassCalibrationError(num_classes=len(config["classes"]))
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]), config)
            loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
            running_loss += loss.item()
            num_instances += len(images)
            ece.update(outputs, torch.argmax(labels, dim=1))
        if reduction == "mean":
            running_loss = running_loss / num_instances
        losses.append(running_loss)
        eces.append(ece.compute().item() * 100)
    eps_sup = max(losses)

    if config["lth"]:
        model = interpolate_model_weights_lth(model_1, model_2, wgs_1=1., wgs_2=1.)
    else:
        assert False
    running_loss = 0.
    num_instances = 0
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]), config)
        loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
        running_loss += loss.item()
        num_instances += len(images)
    if reduction == "mean":
        running_loss = running_loss / num_instances
    eps_sum = running_loss


    losses_m1_varying = []
    eces_m1_varying = []
    for alpha in alphas:
        if config["lth"]:
            model = interpolate_model_weights_lth(model_1, model_2, wgs_1=alpha, wgs_2=1)
        else:
            assert False
        running_loss = 0.
        num_instances = 0
        model.eval()
        ece = MulticlassCalibrationError(num_classes=len(config["classes"]))
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]), config)
            loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
            running_loss += loss.item()
            num_instances += len(images)
            ece.update(outputs, torch.argmax(labels, dim=1))
        if reduction == "mean":
            running_loss = running_loss / num_instances
        losses_m1_varying.append(running_loss)
        eces_m1_varying.append(ece.compute().item() * 100)
    losses_m2_varying = []
    eces_m2_varying = []
    
    for alpha in alphas:
        if config["lth"]:
            model = interpolate_model_weights_lth(model_1, model_2, wgs_1=1, wgs_2=alpha)
        else:
            assert False
        running_loss = 0.
        num_instances = 0
        model.eval()
        ece = MulticlassCalibrationError(num_classes=len(config["classes"]))
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]), config)
            loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
            running_loss += loss.item()
            num_instances += len(images)
            ece.update(outputs, torch.argmax(labels, dim=1))
        if reduction == "mean":
            running_loss = running_loss / num_instances
        losses_m2_varying.append(running_loss)
        eces_m2_varying.append(ece.compute().item() * 100)
    '''
    barrier = eps_sup - eps_bar
    barrier_sum = eps_sum - eps_bar
    front = [x - eps_bar for x in losses]
    losses_m1_varying = [x - eps_bar for x in losses_m1_varying]
    losses_m2_varying = [x - eps_bar for x in losses_m2_varying]
    if relative:
        barrier = (barrier * 100) / eps_bar
        barrier_sum = (barrier_sum * 100) / eps_bar
        front = [(x * 100) / eps_bar for x in front]
        losses_m1_varying = [(x * 100) / eps_bar for x in losses_m1_varying]
        losses_m2_varying = [(x * 100) / eps_bar for x in losses_m2_varying]
    '''
    barrier = eps_sup - eps_bar
    barrier_sum = eps_sum - eps_bar
    front = [x - interp_erorrs[i*2] for i,x in enumerate(losses)]
    losses_m1_varying = [x - interp_erorrs[i] for i,x in enumerate(losses_m1_varying)]
    losses_m2_varying = [x - interp_erorrs[::-1][i] for i,x in enumerate(losses_m2_varying)]
    if relative:
        barrier = (barrier * 100) / eps_bar
        barrier_sum = (barrier_sum * 100) / eps_bar
        front = [(x * 100) / interp_erorrs[i*2] for i,x in enumerate(front)]
        losses_m1_varying = [(x * 100) / interp_erorrs[i] for i,x in enumerate(losses_m1_varying)]
        losses_m2_varying = [(x * 100) / interp_erorrs[::-1][i] for i,x in enumerate(losses_m2_varying)]
    print(f"eps_bar: {eps_bar}, eps_sup: {eps_sup}, barrier: {barrier}, barrier sum: {barrier_sum}")
    print(*front)

    dict_to_export = {"front" : front,
                      "alphas" : alphas,
                      "barrier" : barrier_sum,
                      "relative" : relative,
                      "losses_m1_varying" : losses_m1_varying, 
                      "losses_m2_varying" : losses_m2_varying, 
                      "eces" : eces,
                      "eces_m1_varying" : eces_m1_varying,
                      "eces_m2_varying" : eces_m2_varying
    }
    # specify a filename
    if config["dataset_name"] == "fashion_mnist":
        ds_name = "FMNIST"
    elif config["dataset_name"] == "mnist":
        ds_name = "MNIST"
    else:
        ds_name = config["dataset_name"].replace("_", "")  

    if config["model"]["name"] == "MLP" and len(config["model"]["hidden_dims"]) > 1:
        model_name = "MLPD"
    elif config["model"]["name"] == "MLP" and len(config["model"]["hidden_dims"]) == 1:
        model_name = "MLPS"
    else:
        model_name = "CNN"

        
    filename_dict = f"./pickles_lb/{ds_name}_{seed}_{Path(output).stem[3:]}_{int(config['lth'])}_{model_name}_{chosen_loader}.pickle"
    # write to pickle
    with open(filename_dict, 'wb') as f:
        pickle.dump(dict_to_export, f)
    print(f"Dictionary saved to {filename_dict}")
    
    #plot_front(front, alphas, barrier, barrier_sum, filename=output, relative=relative, losses_m1_varying=losses_m1_varying, losses_m2_varying=losses_m2_varying)
    #plot_front_new(front, alphas, barrier, barrier_sum, filename=output, relative=relative, losses_m1_varying=losses_m1_varying, losses_m2_varying=losses_m2_varying, eces=eces, eces_m1_varying=eces_m1_varying, eces_m2_varying=eces_m2_varying)

    return barrier



def main():
    app()


if __name__ == "__main__":
    app()