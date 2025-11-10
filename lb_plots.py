from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

def plot_front_new(
               alphas,
               filename=None,
               relative=False,
               losses_m1_varying_mean_train=None,
               losses_m2_varying_mean_train=None,
               losses_m1_varying_std_train=None,
               losses_m2_varying_std_train=None,
               losses_m1_varying_mean_test=None,
               losses_m2_varying_mean_test=None,
               losses_m1_varying_std_test=None,
               losses_m2_varying_std_test=None
    ):
    """
    • Builds new_alphas = linspace(0,1,2N-1)
    • Plots `front` only at the original alphas (N points)
    • Builds new_losses = losses_m2_varying + reversed(losses_m1_varying)[1:]
      so it's length 2N-1 and aligns with new_alphas.
    • Adds two extra bottom x-axes: 
      – M2 shows [α0…αN-1, 1…1] 
      – M1 shows [1…1, αN-1…α0]
    """

    plt.rcParams.update({
        'font.size': 14*2,
        'axes.titlesize': 16*2.5,
        'axes.labelsize': 14*2.5,
        'xtick.labelsize': 12*2.5,
        'ytick.labelsize': 12*2.5,
        'legend.fontsize': 12*2.5,
        'figure.titlesize': 16*2.5
    })
    N = len(alphas)
    # 1) make the extended α‐axis
    new_alphas = np.linspace(0, 1, num=2*N - 1, endpoint=True)

    # 2) build the concatenated "varying" losses list
    #    note: losses_m1_varying and losses_m2_varying are Python lists
    new_losses_train = np.concatenate([
        losses_m2_varying_mean_train,
        losses_m1_varying_mean_train[::-1][1:]
    ])
    
    new_losses_std_train = np.concatenate([
        losses_m2_varying_std_train,
        losses_m1_varying_std_train[::-1][1:]
    ])
    new_losses_test = np.concatenate([
        losses_m2_varying_mean_test,
        losses_m1_varying_mean_test[::-1][1:]
    ])
    
    new_losses_std_test = np.concatenate([
        losses_m2_varying_std_test,
        losses_m1_varying_std_test[::-1][1:]
    ])
    # 3) plot
    fig, ax = plt.subplots(figsize=(8,6))

    # ── main front & barrier & optional sum_merge ─────────────

    # ── the combined varying-loss curve ───────────────────────
    ax.plot(new_alphas, new_losses_train,
            '-^',
            label='MaxEntropy Loss train')

    ax.fill_between(
        new_alphas,
        new_losses_train - new_losses_std_train,
        new_losses_train + new_losses_std_train,
        alpha=0.3,            # adjust transparency
        label='±1 σ'
    )
    
    ax.plot(new_alphas, new_losses_test,
            '-^',
            label='MaxEntropy Loss Test')

    ax.fill_between(
        new_alphas,
        new_losses_test - new_losses_std_test,
        new_losses_test + new_losses_std_test,
        alpha=0.3,            # adjust transparency
        label='±1 σ'
    )

    ax.axhline(0, linestyle='--', linewidth=2, color='gray', label='Zero reference')
    
    ax.set_xlabel('t')
    ax.set_ylabel('Loss %' if relative else 'Loss')

    # 7) legend, layout, save/show
    #ax.legend(loc='best')

    if filename:
        # get the directory part of the path
        out_dir = os.path.dirname(filename)
        # only mkdir if there actually is a directory component
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
    plt.show()




def main():

    folder = Path('./pickles_lb/')
    
    records = []
    for pkl in folder.glob('*.pickle'):
        # parse metadata from the filename
        dataset, seed, label, lth, model, split = pkl.stem.split('_')
        
        # load the dict
        with pkl.open('rb') as f:
            content = pickle.load(f)
        
        # merge metadata + content into one flat record
        rec = {
            'dataset': dataset,
            'seed': seed,
            'label': label,
            'lth': lth,
            'model': model,
            'split': split,
            **content
        }
        records.append(rec)
    
    # build your DataFrame
    df = pd.DataFrame(records)
    
    # optionally make a MultiIndex for easy lookup
    df = df.set_index(['dataset','seed','label','lth','model','split'])
    
    # inspect
    #print(df.columns)     # Index(['front','alphas','barrier','relative', ...], dtype=object)
    #print(df.head())

    dataset = df.index.get_level_values('dataset').unique()
    label = df.index.get_level_values('label').unique()
    lth  = df.index.get_level_values('lth').unique()
    model  = df.index.get_level_values('model').unique()


    for dataset_name in sorted(dataset):
        for l in sorted(label):
            for lh in lth:
                for m in model:
                    train = df.xs((dataset_name, l, lh, m, 'train'), level=('dataset','label','lth', 'model', 'split'))
                    test = df.xs((dataset_name, l, lh, m, 'test'), level=('dataset','label','lth', 'model', 'split'))
    
                    if train.size == 0 or test.size == 0:
                        continue
                    
                    losses_m1_varying_train = np.stack(train['losses_m1_varying'].values)
                    losses_m2_varying_train = np.stack(train['losses_m2_varying'].values)
                    losses_m1_varying_train_mean = np.mean(losses_m1_varying_train, axis=0)
                    losses_m1_varying_train_std = np.std(losses_m1_varying_train, axis=0)
                    losses_m2_varying_train_mean = np.mean(losses_m2_varying_train, axis=0)
                    losses_m2_varying_train_std = np.std(losses_m2_varying_train, axis=0)
        
                    test = df.xs((dataset_name, l, lh, m, 'test'), level=('dataset','label','lth', 'model', 'split'))
                    losses_m1_varying_test = np.stack(test['losses_m1_varying'].values)
                    losses_m2_varying_test = np.stack(test['losses_m2_varying'].values)
                    losses_m1_varying_test_mean = np.mean(losses_m1_varying_test, axis=0)
                    losses_m1_varying_test_std = np.std(losses_m1_varying_test, axis=0)
                    losses_m2_varying_test_mean = np.mean(losses_m2_varying_test, axis=0)
                    losses_m2_varying_test_std = np.std(losses_m2_varying_test, axis=0)
                    
                    print(dataset_name, l, model)
                    plot_front_new(alphas=train.alphas.iloc[0], 
                       relative=train.relative.iloc[0], 
                       filename=f'./lb_plots/{dataset_name}_{l}_{lh}_{m}',
                       losses_m1_varying_mean_train=losses_m1_varying_train_mean,
                       losses_m2_varying_mean_train=losses_m2_varying_train_mean,
                       losses_m1_varying_std_train=losses_m1_varying_train_std,
                       losses_m2_varying_std_train=losses_m2_varying_train_std, 
                       losses_m1_varying_mean_test=losses_m1_varying_test_mean,
                       losses_m2_varying_mean_test=losses_m2_varying_test_mean,
                       losses_m1_varying_std_test=losses_m1_varying_test_std,
                       losses_m2_varying_std_test=losses_m2_varying_test_std
                      )





if __name__ == "__main__":
    main()