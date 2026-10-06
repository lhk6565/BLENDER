import os
import re
import wandb
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

api = wandb.Api()

data_list = {'ColoredMNIST': ['0vis7zli', '3d3lz9ob', 'pyot7r9p'],
             'RotatedMNIST': ['0nxgxerj', 'pzaksnei', 'e1w0kpi5', 'agpe6vbn', '122lnr1g', '24vupuii'],
             'VLCS': ['xrwew8lz', 'w87od26q', '2x72u048', 'l77k65ma'],
             'PACS': ['zu03bk59', '87kryhhn', 'a4qibjwi', 'tz6x6id9'],
             'OfficeHome': ['p8e3d2ha', 'twmskfqk', '45nu5beg', 'e6j8wbpj'],
             'DomainNet': ['747qemvu', 'zjaqfgsf', '41fin8bg', 't3y9dcy8', 'q5hk8xsj', 'gmczlown'],
             'TerraIncognita': ['or7qrzuz', 'm4naiszr', 'bdbuofx0', 'au5334ea']}

metrics = ["tr_hsic_inv_spec", "tr_hsic_dinv", "tr_accuracy",
           "val_hsic_inv_spec", "val_hsic_dinv", "val_accuracy",]

splits = [(metrics[0], metrics[1], metrics[2], "Train"),
          (metrics[3], metrics[4], metrics[5], "Validation")]

def get_best_epoch(run):
    last_epoch = None
    pattern = re.compile(r"\[Saved\] best model at epoch (\d+)")
    r = requests.get(run.file("output.log").url, stream=True)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue

        match = pattern.search(line)
        if match:
            last_epoch = int(match.group(1))

    return last_epoch

def plot_hsic_history(history_df, save_path='./analyses/figures/hsic_trajectory'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for ax, (x_col, y_col, c_col, title) in zip(axes, splits):
        x = history_df[x_col].values
        y = history_df[y_col].values
        c = history_df[c_col].values

        # 1) line: epoch trajectory
        ax.plot(x, y, color="gray", linewidth=1, alpha=0.7, zorder=1)

        # 2) scatter: colored by accuracy
        sc = ax.scatter(x, y, c=c, cmap="viridis_r", s=40, zorder=2)

        # 3) highlight: start / end points
        ax.scatter(x[0], y[0], marker="o", color="none", edgecolor="blue", linewidth=1.5, s=120, label="Start", zorder=3)
        ax.scatter(x[-1], y[-1], marker="o", color="none", edgecolor="red", linewidth=1.5, s=120, label="End", zorder=3)

        ax.set_xlabel(r"$\mathrm{HSIC}(z_{\mathrm{inv}}, z_{\mathrm{spec}})$", fontsize=14)
        ax.set_ylabel(r"$\mathrm{HSIC}(d, z_{\mathrm{inv}})$", fontsize=14)

        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='lower right', fontsize=10)
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label("Accuracy", fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/hsic_{dataset_name}_{run_idx}.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        plt.show()


for dataset_name, run_ids in tqdm(data_list.items(), desc='Datasets', leave=False):
    for run_idx, run_id in enumerate(run_ids):
        run = api.run(f'/BLENDER_paper/blender_{dataset_name}/runs/{run_id}')
        history = run.history(keys=metrics)
        best_epoch = get_best_epoch(run)
        history = history[history["_step"] <= best_epoch]

        plot_hsic_history(history, save_path='./analyses/figures/hsic_trajectory')