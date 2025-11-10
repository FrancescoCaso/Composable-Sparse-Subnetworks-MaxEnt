import argparse
import pandas as pd
import numpy as np
from hashlib import sha256

import torch
from pathlib import Path
from tqdm import tqdm
from itertools import product, combinations, permutations
from frankenstein_sample_level import single_evaluation_statistics, single_comparison_statistics, incremental_comparison_statistics
import yaml
from tqdm.contrib.concurrent import thread_map
from dataset_creation import str2bool
from sqlalchemy import create_engine
engine = create_engine('sqlite:///models_results.db', echo=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate results tables.')
    parser.add_argument('-r', '--results_section', type=int, required=True, help='Section of the results to generate: 1 for single class, 2 for comparison.')
    parser.add_argument('-s', '--seeds', type=str, default='0,1,2,3,4', help='Comma-separated list of seeds to use for training. If multiple seeds are provided, they will be used in a loop.')
    parser.add_argument('-m', '--model_types', type=str, default='shallow_mlp,deep_mlp,cnn', help='Comma-separated list of model types to use for training.')
    parser.add_argument('-d', '--datasets', type=str, default='mnist,fmnist,human_activity_recognition', help='Comma-separated list of datasets to use for training.')
    parser.add_argument('--comb_file', type=str, default=None, help='Path to the file containing the combinations of classes to use for evaluation.')
    parser.add_argument('--comb_idx', type=int, default=None, help='Index of the combinations batch to use for evaluation.')
    parser.add_argument('--cardinality', type=str, default='1,2,5', help='Comma-separated list of cardinalities to use for evaluation of results 2.')
    parser.add_argument('--nolth', type=str2bool, default=False, help='Get experimental results without LTH.')
    print("aaa")

    return parser



def main() -> None:
    print("boh")
    args = build_parser().parse_args()
    print(args)

    seeds = [int(s) for s in args.seeds.split(',')]
    model_types = args.model_types.split(',')
    datasets = args.datasets.split(',')
    
    # model_types = ['deep_mlp', 'shallow_mlp']
    # datasets = ['mnist', 'fmnist']#, 'human_activity_recognition']
    # datasets = ['human_activity_recognition']
    # classes = range(10)
    
    np.random.seed(0)
    
    if args.results_section == 1:
        def idx_hash(row):
            # hash for class, dataset, model, and lth
            return sha256(f"{row['Class']}_{row['dataset']}_{row['model']}_{row['lth']}".encode()).hexdigest()
        
        combs = []
        for m in model_types:
            for ds in datasets:
                with open(next(Path('configs').rglob(f'setting_{m}_{ds}{"_nolth" if args.nolth else ""}.yaml')), 'r') as f:
                    classes = yaml.safe_load(f)['classes']
                combs += list(product([m1 for m1 in [m] if m != 'cnn' or ds in ['mnist', 'fmnist']], 
                                      [ds], classes))
    
        def single_comp(mt_ds_cl):
            mt, ds, cl = mt_ds_cl
    
            base_conf_path = (Path('configs')/f'setting_{mt}_{ds}{"_nolth" if args.nolth else ""}.yaml')
            assert base_conf_path.exists(), f"Config file {base_conf_path} does not exist"
            with open(base_conf_path) as f:
                this_config = yaml.safe_load(f)
    
            exp_hash = idx_hash({'Class': cl, 'dataset': ds, 'model': mt, 'lth': this_config['lth']})
            try:
                df = pd.read_sql(f"SELECT * FROM results{args.results_section} WHERE hash='{exp_hash}'", con=engine)
                if len(df) > 0:
                    return df
            except Exception as e:
                pass
            
            assert cl in this_config['classes'], f"Class {cl} not in config {this_config['classes']}"
                
            # print(f"Processing {mt}, {ds}, {cl}")
    
            df = single_evaluation_statistics(
                config=this_config, class1=cl, 
                output=None, aligned=False, seeds=seeds, is_val=False
            ).assign(Class=cl, dataset=ds, model=mt, lth=this_config['lth'])
    
            
            df['hash'] = df.apply(idx_hash, axis=1)
            # insert into db
            df.to_sql(
                f"results{args.results_section}",
                con=engine,
                if_exists='append',
                index=False, 
                index_label='hash',)
            return df
        # table with schema
        # seed | accuracy | rewarded_accuracy | entropy | Class | dataset | model | lth
    
    elif args.results_section == 2:
        def idx_hash(row):
            # hash for classes1, classes2, dataset, model, and lth
            return sha256(f"{row['Class1']}_{row['Class2']}_{row['dataset']}_{row['model']}_{row['lth']}".encode()).hexdigest()
    
        class_cardinalities = [int(c) for c in args.cardinality.split(',')]
    
        if args.comb_file is None:
    
            combs = []
            for m in model_types:
                for ds in datasets:
                    with open(next(Path('configs').rglob(f'setting_{m}_{ds}{"_nolth" if args.nolth else ""}.yaml')), 'r') as f:
                        print(f)
                        classes = yaml.safe_load(f)['classes']
    
                    cards_combs = []
                    for card in class_cardinalities:
                        if 'mixed' in ds:
                            current_combs = [a + b for a, b in product(combinations(classes[0], card), combinations(classes[1], card))]
                        else:
                            current_combs = list(combinations(classes, card*2)) # separates into model a and model b
                        print(ds, classes, card, list(combinations(classes, card*2)))
                        if len(current_combs) > 10:
                            comb_idx = np.random.choice(len(current_combs), size=10, replace=False)
                            current_combs = [current_combs[i] for i in comb_idx]
                        cards_combs.extend(current_combs)
                    combs += list(product([m1 for m1 in [m] if m != 'cnn' or ds in ['mnist', 'fmnist']],
                                        [ds], cards_combs))
                

        else:
            if any([ds not in ['mnist', 'fmnist'] for ds in datasets]):
                raise NotImplementedError()
            classes = list(range(10))
            if args.comb_idx is None:
                raise ValueError("comb_idx must be provided if comb_file is provided")
            
            with open(args.comb_file, 'r') as f: # yaml file with batched class combinations
                cards_combs = [tuple(yaml.safe_load(f)[args.comb_idx])]
            
            combs = []
            for ds in datasets:
                combs += list(product(model_types, [ds], cards_combs))
    
        def single_comp(mt_ds_cl):
            mt, ds, cl = mt_ds_cl
    
            base_conf_path = (Path('configs')/f'setting_{mt}_{ds}{"_nolth" if args.nolth else ""}.yaml')
            assert base_conf_path.exists(), f"Config file {base_conf_path} does not exist"
            with open(base_conf_path) as f:
                this_config = yaml.safe_load(f)
    
            
            exp_hash = idx_hash({'Class1': cl[:len(cl)//2], 'Class2': cl[len(cl)//2:], 'dataset': ds, 'model': mt, 'lth': this_config['lth']})
            try:
                df = pd.read_sql(f"SELECT * FROM results{args.results_section} WHERE hash='{exp_hash}'", con=engine)
                if len(df) > 0:
                    return df
            except Exception as e:
                pass
                
            classes1, classes2 = cl[:len(cl)//2], cl[len(cl)//2:]
            df = single_comparison_statistics(
                config=this_config, output=None, layers=None, calc_loss_barrier=False,
                class1=classes1, class2=classes2, aligned=False, seeds=seeds, is_val=False
            ).assign(dataset=ds, model=mt, cardinality=len(classes1), lth=this_config['lth'])
            df['Class1'] = ','.join([str(c) for c in classes1])
            df['Class2'] = ','.join([str(c) for c in classes2])
            df['hash'] = df.apply(idx_hash, axis=1)
            # insert into db
            df.to_sql(
                f"results{args.results_section}",
                con=engine,
                if_exists='append',
                index=False, 
                index_label='hash',)
            return df
            # table with schema
            # seed | accuracy | rewarded_accuracy | entropy | Class1 | Class2 | dataset | model | cardinality | lth
    
    elif args.results_section == 3:
        def idx_hash(row):
            # hash for permutations, dataset, model, and lth
            return sha256(f"{row['Permutation']}_{row['dataset']}_{row['model']}_{row['lth']}".encode()).hexdigest()
        
        combs = []
        for m in model_types:
            for ds in datasets:
                with open(next(Path('configs').rglob(f'setting_{m}_{ds}{"_nolth" if args.nolth else ""}.yaml')), 'r') as f:
                    classes = yaml.safe_load(f)['classes']
                    if 'mixed' in ds:
                        classes = sum(classes, [])
                # sample 10 random indices
                if len(classes) > 3:
                    perms_classes = []
                    for _ in range(10):
                        np.random.shuffle(classes)
                        perms_classes.append(tuple(classes.copy()))
                else:
                    perms_classes = permutations(classes, len(classes))
                combs += list(product([m1 for m1 in [m] if m != 'cnn' or ds in ['mnist', 'fmnist']],
                                      [ds], perms_classes))
            
        def single_comp(mt_ds_cl):
            mt, ds, cl = mt_ds_cl
            base_conf_path = (Path('configs')/f'setting_{mt}_{ds}{"_nolth" if args.nolth else ""}.yaml')
            assert base_conf_path.exists(), f"Config file {base_conf_path} does not exist"
            with open(base_conf_path) as f:
                this_config = yaml.safe_load(f)
    
            exp_hash = idx_hash({'Permutation': cl, 'dataset': ds, 'model': mt, 'lth': this_config['lth']})
            try:
                df = pd.read_sql(f"SELECT * FROM results{args.results_section} WHERE hash='{exp_hash}'", con=engine)
                if len(df) > 0:
                    return df
            except Exception as e:
                pass
    
            df = incremental_comparison_statistics(
                    config=this_config, classes=cl, aligned=False, 
                    output=None, calc_loss_barrier=False,
                    seeds=seeds, is_val=False
                ).assign(dataset=ds, model=mt, lth=this_config['lth'])
            df['Permutation'] = ','.join([str(c) for c in cl])
            df['hash'] = df.apply(idx_hash, axis=1)
            # insert into db
            df.to_sql(
                f"results{args.results_section}",
                con=engine,
                if_exists='append',
                index=False, 
                index_label='hash',)
            return df
            # table with schema
            # seed | accuracy | rewarded_accuracy | entropy | Permutation | level | dataset | model
    
    
    # all_dfs = thread_map(
    #         lambda c: single_comp(c),
    #         combs,
    #         max_workers=1,
    #         total=len(model_types)*len(datasets)*len(classes),
    #         desc="Processing",
    #     )

    all_dfs = []
    print(combs)
    for mt, ds, cl in tqdm(combs):
        all_dfs.append(single_comp((mt, ds, cl)))


if __name__ == "__main__":
    main()
