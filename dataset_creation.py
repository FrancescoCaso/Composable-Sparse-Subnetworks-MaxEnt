import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from models import *
from train_utils import *
from log import *
import yaml
import copy
from lottery_dataset_creation import main_lth, convert_to_soft_targets, convert_to_anomaly_det_targets, generate_class_masks
from models_db import ModelsDB
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def str2bool(v):
    """
    Convert a string to a boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
        parser = argparse.ArgumentParser(description='Train an MLP on MNIST with specified classes.')
        parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
        parser.add_argument('--dataset_name_custom', type=str, default=None, help='Replace the dataset in config with another one.')
        parser.add_argument('--lth_custom', type=str2bool, default=None, help='Replace the dataset in config with another one.')
        parser.add_argument('--n_classes', type=int, default=1, help='Number of classes ech dummy model should know.')
        parser.add_argument('--classes', type=str, default=None, help='List of classes to use as rewarded classes for training.')
        parser.add_argument('--seed', type=str, required=False, default='0,1,2,3,4', help='Seed or list of seeds to use for training. If multiple seeds are provided, they will be used in a loop.')
        return parser.parse_args()

def main(config, device, args, rw=None, db=None):
    if isinstance(rw, int) and rw is not None:
        rw = [rw]
    if db is None:
        db = ModelsDB(refresh=False)
    
    # config["seeds"] = np.arange(config["seeds"])
    config["seed"] = args.seed

    # anomaly_flag = (config["loss_name"] == "one_vs_all")
    model_name = config["model"]["name"]

    if model_name == 'MLP':
        params = {
            "input_dim": config["model"]["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "output_dim": len(config["classes"]),
            "dropout_rate": config["model"]["dropout"],
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
            continue

        epochs = config["epochs"]
        config["weights"] = comb

        
            # config["seed"] = seed
        logger = initialize_logger_from_config(config)
        logger.log(config)
        
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = create_model(config['model']['name'], params)
        model.to(device)
        class_probs = {np.arange(len(config["classes"]))[i]:config["weights"][i] for i in range(len(config["classes"]))}
        # criterion = CrossEntropyLoss()
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        # criterion = loss_fn(F.log_softmax(pred_logits, dim=1), target_probs)
        optimizer = get_optimizer(model.parameters(), **config['optimizer'])
        best_val_loss = float('inf')
        best_model = None
        best_epoch = 0
        patience = 3
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_labels = []
            total_pred = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                y = copy.deepcopy(labels)
                # if anomaly_flag:
                #     labels = convert_to_anomaly_det_targets(labels, class_probs, len(config["classes"]))
                # else:
                labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]))
                loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
                # loss = loss_fn(outputs.log(), labels)
                # loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_labels.append(y)
                total_pred.append(predicted)
            total_pred = torch.cat(total_pred, dim=0)
            total_labels = torch.cat(total_labels, dim=0)
            accuracy = (total_labels == total_pred).sum().item() / len(total_labels)
            logger({'epoch':epoch, 'train_accuracy':accuracy})
            # validate
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                total_labels = []
                total_pred = []
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    y = copy.deepcopy(labels)
                    labels = convert_to_soft_targets(labels, class_probs, len(config["classes"]))
                    loss = loss_fn(F.log_softmax(outputs, dim=1), labels)
                    # loss = loss_fn(outputs.log(), labels)
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
        
        # Save the best model
        model = best_model
        config['model_path'] = (logger.results_directory / f'{logger.name}.pt').as_posix()
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': best_epoch,
                    }, config['model_path'])
        accuracy, cm, rewarded_accuracy, mean_non_rewarded_entropy = eval(model, test_loader, rewarded_classes=rewarded_classes, device=device)
        logger({'test_accuracy':accuracy, 'cost_matrix':cm, 'rewarded_accuracy':rewarded_accuracy, 'mean_non_rewarded_entropy':mean_non_rewarded_entropy})
        logger.finish()
        return config, logger.name

 
def create_from_config(config, seed, device='cpu', 
                       args=argparse.Namespace(n_classes=1), 
                       rewarded_classes=None,
                       db=None):
    args.seed = seed  
    my_config = copy.deepcopy(config)
    if config['lth']:
        out = main_lth(my_config, device, args, rw=rewarded_classes, db=db)
    else:
        out = main(my_config, device, args, rw=rewarded_classes, db=db)
    
    if out is not None:
        exp_config, hash = out
        db.add(exp_config, hash)
    else:
        exp_config = None
    return [exp_config]
  

def retrieve_model(config, aligned: bool=False, device: str = 'cuda'):
    rewarded_class = config.get('rewarded_classes', '')
    if len(rewarded_class) > 1:
        config_id = config['file_id'].split('_')[0]
    else:
        config_id = config['file_id']
    model_directory = Path(config['model_path'])
    if aligned:
        model_directory = model_directory.with_name(model_directory.stem + '_aligned.pt')

    if not config["model"]["name"] in ['MLP', 'ConvNet']:
        raise ValueError(f"Model {config['model']['name']} not supported.")

    params = config["model"]
    params['activation_fn'] = getattr(nn, config["model"]["activation_fn"])
    params['output_dim'] = len(config["classes"])
    # model = load_model(params = params, model_ = model_directory, model_name = new_config['model']['name'])
    
    model = create_model(config["model"]["name"], params)
    if config["lth"]:
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                prune.custom_from_mask(module, name='weight', mask=torch.ones_like(module.weight))
            if hasattr(module, 'bias') and module.bias is not None:
                prune.custom_from_mask(module, name='bias', mask=torch.ones_like(module.bias))

    if type(model_directory) in [str, PosixPath]:
        model_dict = torch.load(model_directory, weights_only=False, map_location=device)['model_state_dict']
    elif type(model_dict) != dict:
        raise ValueError("model_ must be a dictionary or a path to a model.")

    model.load_state_dict(model_dict)
    model.to(device)

    return model, str(rewarded_class)


def get_model_config(config, seed, rewarded_classes, db=None):
    config['classes'] = ClassesList(config["classes"], config["dataset_name"])
    dataset_name = config['classes'].get_dataset(rewarded_classes) if config['classes'].is_multidataset else config['dataset_name']

    results = db.query(config, seed, rewarded_classes, dataset_name=dataset_name)
    if len(results) == 0:
        # print(f"Cannot find any model") # TODO: fixZ
        # return None
        if config['classes'].is_multidataset:
            raise ValueError(f"Can regenerate only for a single dataset.")
        print(f"Cannot find any model (dims: {config['model'].get('hidden_dims')}) for classes {rewarded_classes} with seed {seed}. Generating a new one.")
        results = create_from_config(config, seed=seed, rewarded_classes=rewarded_classes, db=db, device=device)

    new_config = copy.deepcopy(results[0])

    new_config["classes"] = ClassesList(new_config["classes"], new_config["dataset_name"])
    return new_config

    
#### LEGACY ####
def retrieve_model_directory(config: dict, aligned: bool = False):
    """
    It retrieves a model directory according to a specific configuration.
    """
    str_classes = ''.join(map(str, config["classes"]))
    choosen_class = ''.join(map(str, [config["classes"][ind] for ind, val in enumerate(config["weights"]) if val==1]))
    results_directory = Path('./lth_results' if config["lth"] else './results') 

    if not getattr(config['classes'], 'is_multidataset', False):
        dataset_name = config["dataset_name"]
        hash = f"{generate_config_hash(config)}"
    else:
        # TODO: Al momento assumo che str_classes sia sempre la stessa ('0~num_classes' per tutti i dataset)
        dataset_name = [config["classes"].get_dataset_name(cl) for cl in choosen_class]
        assert len(set(dataset_name)) == 1, f"Dataset names are not the same for all classes: {dataset_name}"
        dataset_name = dataset_name[0]
        single_ds_config = copy.deepcopy(config)
        single_ds_config["dataset_name"] = dataset_name
        single_ds_config["classes"] = [cl for cl in config["classes"]]
        hash = f"{generate_config_hash(single_ds_config)}"

    if aligned:
        print('aligned models')
        hash = hash + '_aligned.pt'
    else:
        hash = hash + '.pt'
    results_directory = results_directory/ dataset_name / str_classes / choosen_class / hash
    if not results_directory.exists():
        print(f"Cannot find any model at path {results_directory}")
        # check with similars
        for file in results_directory.parent.glob('*.yaml'):
            print(f" - Found {file.name}")
            with open(file, 'r') as f:
                other_config = yaml.safe_load(f)
            curr_conf = config if not getattr(config['classes'], 'is_multidataset', False) else single_ds_config
            for key in curr_conf.keys():
                if key in other_config and curr_conf[key] != other_config[key]:
                    print(f"\t - Different {key} between {curr_conf[key]} and {other_config[key]}")
        raise ValueError(f"Cannot find any model at path {results_directory}")

    return results_directory

def retrieve_model_old(config: dict, rewarded_class: int = None, aligned: bool=False, device: str = 'cuda'):
    """
    This function returns the model according to a specific configuration and rewarded class.
    If rewarded_class is None, then it returns the model without any specific reward (standard model).
    The configuration is intended the base configuration, still without seed and weights of the classes.
    The full configuration will be retieved by retrieve model directory.
    """
    warnings.warn("🚨🚨🚨 retrieve_model could become deprecated, try using retrieve_model_new instead", DeprecationWarning)
    print(rewarded_class)
    title = ''
    # for classes in config['classes']:
    #     name += str(classes)
    if rewarded_class is not None:
        title+=f'{rewarded_class}'
    
    label_mapping = {original: new for new, original in enumerate(config["classes"])} # to be used since the classes are not naturally ordered
    n_classes = len(config['classes'])
    new_config = copy.deepcopy(config)
    if rewarded_class is not None:
        new_config['weights'] =  [1 if i == label_mapping[rewarded_class] else 1/n_classes for i in range(n_classes)]
    else:
        print('here')
        new_config['weights'] =  [1] * n_classes
    new_config['seed'] = 0 #TODO: choose a standard for the seed
    
    model_directory = retrieve_model_directory(new_config, aligned)
    print(model_directory)

    if not config["model"]["name"] in ['MLP', 'ConvNet']:
        raise ValueError(f"Model {config['model']['name']} not supported.")

    params = config["model"]
    params['activation_fn'] = getattr(nn, config["model"]["activation_fn"])
    params['output_dim'] = len(config["classes"])

    # model = load_model(params = params, model_ = model_directory, model_name = new_config['model']['name'])
    model_ = model_directory
    model = create_model(new_config["model"]["name"], params)
    if config["lth"]:
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                prune.custom_from_mask(module, name='weight', mask=torch.ones_like(module.weight))
            if hasattr(module, 'bias') and module.bias is not None:
                prune.custom_from_mask(module, name='bias', mask=torch.ones_like(module.bias))
    
    if type(model_) in [str, PosixPath]:
        model_ = torch.load(model_, weights_only=False, map_location=device)['model_state_dict']
    elif type(model_) != dict:
        raise ValueError("model_ must be a dictionary or a path to a model.")
    
    model.load_state_dict(model_)
    model.to(device)
    return model, title

    


if __name__ == "__main__":
    db = ModelsDB()
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for arg in vars(args):
        if arg.endswith('_custom') and getattr(args, arg) is not None:
            config[arg.split('_custom')[0]] = getattr(args, arg)

    seeds = args.seed.split(',')
    if len(seeds) > 1:
        seeds = [int(s) for s in seeds]
    else:
        seeds = [int(args.seed)]

    for seed in seeds:
        print(f"Running with seed {seed}")
        if args.classes is not None:
            rw_classes = [int(i) for i in args.classes.split(',')]
            if args.n_classes > 1: # split into blocks of n_classes
                assert len(rw_classes) % args.n_classes == 0
                rw_classes = [rw_classes[i:i + args.n_classes] for i in range(0, len(rw_classes), args.n_classes)]
            for rwc in rw_classes:
                print(f"Running with classes {rwc}")
                create_from_config(config, seed, device=device, args=args, rewarded_classes=rwc, db=db)
        else:
            create_from_config(config, seed, device=device, args=args, db=db)
        # args.seed = seed  
        # if config['lth']:
        #     main_lth(config, device, args, db=db)
        # else:
        #     main(config, device, args, db=db)