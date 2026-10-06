import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from functools import lru_cache
from datasets.dataset import *
from utils import *
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable



class LatentVisualizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_class(mod_name: str, cls_name: str):
        module = importlib.import_module(mod_name)
        return getattr(module, cls_name)

    @staticmethod
    def _standardize_encoding(model_name, outputs):
        if model_name.startswith("blender"):
            enc_list, (x_list, y_list, d_list) = outputs
            z_inv_list = []
            z_spec_list = []
            for i in range(len(enc_list)):
                z_inv_list.append(enc_list[i][0])
                z_spec_list.append(enc_list[i][1])
            z_inv, z_spec = torch.cat(z_inv_list, dim=0), torch.cat(z_spec_list, dim=0)
            x, y, d = torch.cat(x_list, dim=0), torch.cat(y_list, dim=0), torch.cat(d_list, dim=0)

            data_cols = [x.flatten(1), z_inv, z_spec]
            col_labels = ["Original", "Invariant Latent", "Specific Latent"]

        elif model_name == "diva":
            enc_list, (x_list, y_list, d_list) = outputs
            z_d_list, z_x_list, z_y_list = [], [], []
            for i in range(len(enc_list)):
                z_d_list.append(enc_list[i][0])
                z_x_list.append(enc_list[i][1])
                z_y_list.append(enc_list[i][2])
            z_d, z_x, z_y = torch.cat(z_d_list, dim=0), torch.cat(z_x_list, dim=0), torch.cat(z_y_list, dim=0)
            x, y, d = torch.cat(x_list, dim=0), torch.cat(y_list, dim=0), torch.cat(d_list, dim=0)
            data_cols = [x.flatten(1), z_d, z_x, z_y]
            col_labels = ["x", "z_d", "z_x", "z_y"]

        return data_cols, col_labels, (y, d)
    
    # initial plot
    @staticmethod
    def plot_latent(dataset_name, model_name, test_domain=[5], save_path='analyses/figures/latentspace/'):
        # Load model
        if model_name in MODEL_DISPATCH:
            builder_name = MODEL_DISPATCH[model_name]

            cfg = ModelConfig(model_name)
            loader_config, model_config, train_config = cfg.resolve(dataset_name)
            model_config['model_name'] = model_name
            dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

            ModelLoaderClass = LatentVisualizer._get_class('methods.builder', builder_name)
            model_loader = ModelLoaderClass(dataloader, model_config, train_config)
            model_loader.model.to(model_loader.device)

            load_model(model_loader.model, model_loader.model_path)
            model_loader.model.eval()

        outputs = model_loader.encode(model_loader.train_loader)
        data_cols, col_labels, (y, d) = LatentVisualizer._standardize_encoding(model_name, outputs)

        # Labels (integers) -> numpy
        y_np = y.reshape(-1).detach().cpu().numpy().astype(int)
        d_np = d.reshape(-1).detach().cpu().numpy().astype(int)

        # Optional subsample
        n = len(y_np)
        if n > 2000:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, 2000, replace=False)
            y_np = y_np[idx]
            d_np = d_np[idx]
            data_cols = [c[idx] for c in data_cols]

        # For clean colorbar ticks
        y_vals = np.unique(y_np)
        d_vals = np.unique(d_np)
        y_len = len(y_vals)
        d_len = len(d_vals)
        
        # 2D embedding once per latent
        emb_2d_list = []
        for c in data_cols:
            z = c.detach().cpu().numpy()
            if z.shape[1] > 2:
                z2 = TSNE(n_components=2, perplexity=100, random_state=get_global_seed()).fit_transform(z)
            else:
                z2 = z
            emb_2d_list.append(z2)

        # Plot
        ncols = len(emb_2d_list)
        fig, axes = plt.subplots(2, ncols, figsize=(5.6 * ncols, 9.2), sharex='col', sharey='col', constrained_layout=True)
        cmap_d = plt.get_cmap('tab10', d_len)
        cmap_y = plt.get_cmap('tab10', y_len)

        if ncols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for i, z2 in enumerate(emb_2d_list):
            ax0, ax1 = axes[0, i], axes[1, i]

            sc0 = ax0.scatter(z2[:, 0], z2[:, 1], c=d_np, cmap=cmap_d, s=14, alpha=0.75, linewidths=0)
            sc1 = ax1.scatter(z2[:, 0], z2[:, 1], c=y_np, cmap=cmap_y, s=14, alpha=0.75, linewidths=0)

            ax0.set_title(col_labels[i], fontsize=13, pad=10)
            ax1.set_xlabel("Dim 1", fontsize=11)
            if i == 0:
                ax0.set_ylabel("Dim 2", fontsize=11)
                ax1.set_ylabel("Dim 2", fontsize=11)

            for ax in (ax0, ax1):
                ax.grid(True, alpha=0.18, linewidth=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        fig.suptitle(f"Latent Space Visualization — {model_name} | {dataset_name} | Test Domain: {test_domain[0]}", fontsize=15)

        # Colorbars (ticks are integer labels)
        cbar0 = fig.colorbar(sc0, ax=axes[0, :], shrink=0.95, pad=0.01, aspect=30)
        cbar0.set_label("Domain", fontsize=11)
        cbar0.set_ticks(d_vals)

        cbar1 = fig.colorbar(sc1, ax=axes[1, :], shrink=0.95, pad=0.01, aspect=30)
        cbar1.set_label("Class", fontsize=11)
        cbar1.set_ticks(y_vals)

        # Save / show
        if save_path:
            os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
            save_path = os.path.join(save_path, model_name, f'{dataset_name}_{test_domain}.pdf')
            plt.savefig(save_path)
            print(f'Latent space figure saved to {save_path}')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    # For final RotatedMNIST specific latent space figures, fix colors per domain
    @staticmethod
    def _fixed_domain_colors_rotatedmnist(rgb_list=None):
        if rgb_list is None:
            rgb_list = [(220, 20, 60),  # red
                        (255, 140, 0),  # orange
                        (255, 214, 0),  # yellow
                        (0, 200, 83),   # green
                        (0, 114, 255),  # blue
                        (156, 39, 176)] # purple

        rgb_array = np.array(rgb_list) / 255.0
        domain_to_color = {d: tuple(rgb_array[d]) for d in range(len(rgb_array))}
        return domain_to_color

    @staticmethod
    def _save_domain_legend_rotatedmnist(dom2col, save_path, model_name="blender", filename="RotatedMNIST_domain_legend.png", domain_labels=None, domain_order=None, dpi=1200):
        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
        out = os.path.join(save_path, model_name, filename)

        if domain_order is None:
            domain_order = list(range(6))

        def _get_label(d: int) -> str:
            if domain_labels is None:
                return f"Domain {d}"
            if isinstance(domain_labels, (list, tuple)):
                return str(domain_labels[d])
            if isinstance(domain_labels, dict):
                return str(domain_labels.get(d, f"Domain {d}"))
            raise TypeError("domain_labels must be None, list/tuple, or dict.")

        handles, labels = [], []
        for d in domain_order:
            handles.append(Line2D([0], [0], marker='o',
                                  linestyle='None',
                                  markersize=10,
                                  markerfacecolor=dom2col[d],
                                  markeredgewidth=0,
                                  color=dom2col[d]))
            labels.append(_get_label(d))

        fig = plt.figure(figsize=(2.6, 4.2), frameon=False)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        ax.legend(handles, labels, loc="center", frameon=False, ncol=6, handletextpad=0.6, labelspacing=0.9, borderpad=0.0, fontsize=11)
        plt.tight_layout()
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        plt.close(fig)
        print(f"[Saved legend] {out}")

    @staticmethod
    def _save_vertical_colorbar_legend(values, colors, label_name, save_path, model_name="blender", filename="legend.png", height=5.0, dpi=1200):
        """
        Save a discrete vertical colorbar legend with fixed width.
        """

        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
        out = os.path.join(save_path, model_name, filename)

        values = list(values)
        n = len(values)

        cmap = ListedColormap(colors)
        bounds = np.arange(n + 1) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig = plt.figure(figsize=(1.3, height), frameon=False)

        ax = fig.add_axes([0.35, 0.05, 0.30, 0.9])

        cbar = fig.colorbar(sm, cax=ax, boundaries=bounds, ticks=np.arange(n), spacing="proportional")

        cbar.set_ticklabels([''] * n)

        fig.savefig(out, bbox_inches="tight", pad_inches=0.03, dpi=dpi)
        plt.close(fig)

        print(f"[Saved {label_name.lower()} colorbar legend] {out}")

    @staticmethod
    def _plot_single_element_tsne(z2, point_colors, out_path, figsize=(5.6, 4.6), point_size=14, alpha=0.75, dpi=600):
        """
        Save a single t-SNE panel with axes/grid/spines, but without titles/labels.
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(z2[:, 0], z2[:, 1], c=point_colors, s=point_size, alpha=alpha, linewidths=0)

        # grid
        ax.grid(True, alpha=0.18, linewidth=0.6, )

        # show all spines
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)

        # no axis labels / titles
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
        plt.close(fig)

    @staticmethod
    def save_all_latents_tsne_rotatedmnist(model_name="blender", test_domains=None, save_path="analyses/figures/latentspace_elements/", max_points=2000, perplexity=100, domain_labels=None, class_labels=None):
        dataset_name = "RotatedMNIST"
        if test_domains is None:
            test_domains = list(range(6))

        if not model_name.startswith("blender"):
            raise ValueError("This function currently assumes blender-style outputs: [Original, Invariant Latent, Specific Latent].")

        class_legend_saved = False
        class_to_color = None

        # Match the per-panel aspect more closely to plot_latent
        panel_figsize = (5.6, 4.6)

        for td in test_domains:
            test_domain = [td]

            dom2col = LatentVisualizer._fixed_domain_colors_rotatedmnist()
            train_domain_order = [d for d in range(6) if d != td]
            train_domain_colors_for_bar = [dom2col[d] for d in train_domain_order]

            LatentVisualizer._save_vertical_colorbar_legend(values=train_domain_order, colors=train_domain_colors_for_bar, label_name="Domain",
                                                            save_path=os.path.join(save_path, model_name, f"td{td}"), model_name="", filename=f"RotatedMNIST_td{td}_domain_colorbar.png", height=4.8, dpi=1200)

            if model_name not in MODEL_DISPATCH:
                raise ValueError(f"Unknown model_name={model_name}. Not in MODEL_DISPATCH.")
            builder_name = MODEL_DISPATCH[model_name]

            cfg = ModelConfig(model_name)
            loader_config, model_config, train_config = cfg.resolve(dataset_name)
            model_config["model_name"] = model_name
            loader_config["num_workers"] = 0

            dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

            ModelLoaderClass = LatentVisualizer._get_class("methods.builder", builder_name)
            model_loader = ModelLoaderClass(dataloader, model_config, train_config)
            model_loader.model.to(model_loader.device)

            load_model(model_loader.model, model_loader.model_path)
            model_loader.model.eval()

            outputs = model_loader.encode(model_loader.train_loader)
            data_cols, col_labels, (y, d) = LatentVisualizer._standardize_encoding(model_name, outputs)

            d_np = d.reshape(-1).detach().cpu().numpy().astype(int)
            y_np = y.reshape(-1).detach().cpu().numpy().astype(int)

            # subsample
            n = len(d_np)
            if n > max_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, max_points, replace=False)
                d_np = d_np[idx]
                y_np = y_np[idx]
                data_cols = [c[idx] for c in data_cols]

            # Save class legend once, as vertical colorbar
            if not class_legend_saved:
                class_vals = np.unique(y_np)
                K = len(class_vals)
                cmap_y = plt.get_cmap("tab10", K)
                class_to_color = {int(c): cmap_y(i) for i, c in enumerate(class_vals)}

                print("[RotatedMNIST class color mapping]")
                for c in class_vals:
                    print(f"  class {int(c)}: {mcolors.to_hex(class_to_color[int(c)])}")

                class_colors_for_bar = [class_to_color[int(c)] for c in class_vals]

                LatentVisualizer._save_vertical_colorbar_legend(
                    values=[int(c) for c in class_vals],
                    colors=class_colors_for_bar,
                    label_name="Class",
                    save_path=save_path,
                    model_name=model_name,
                    filename="RotatedMNIST_class_colorbar.png",
                    height=6.0)
                class_legend_saved = True

            # LODO domain index fix
            # train domains are reindexed 0..4, so restore original domain ids
            d_fix = d_np.copy()
            d_fix[d_fix >= td] += 1

            domain_colors = np.array([dom2col[int(di)] for di in d_fix])
            class_colors = np.array([class_to_color[int(ci)] for ci in y_np])

            os.makedirs(os.path.join(save_path, model_name, f"td{td}"), exist_ok=True)

            for col, name in zip(data_cols, col_labels):
                z = col.detach().cpu().numpy()
                if z.shape[1] > 2:
                    z2 = TSNE(n_components=2, perplexity=perplexity, random_state=get_global_seed()).fit_transform(z)
                else:
                    z2 = z

                safe_name = name.lower().replace(" ", "_")

                # domain-colored panel
                out_domain = os.path.join(save_path, model_name, f"td{td}", f"{dataset_name}_td{td}_{safe_name}_domain.png")
                LatentVisualizer._plot_single_element_tsne(z2=z2, point_colors=domain_colors, out_path=out_domain, figsize=panel_figsize, point_size=14, alpha=0.75, dpi=600)

                # class-colored panel
                out_class = os.path.join(save_path, model_name, f"td{td}", f"{dataset_name}_td{td}_{safe_name}_class.png")
                LatentVisualizer._plot_single_element_tsne(
                    z2=z2,
                    point_colors=class_colors,
                    out_path=out_class,
                    figsize=panel_figsize,
                    point_size=14,
                    alpha=0.75,
                    dpi=600,
                )

            print(f"[Done td={td}] saved original/inv/spec for domain+class coloring")

    @staticmethod
    def save_specific_latent_tsne_rotatedmnist(model_name="blender", test_domains=None, save_path="analyses/figures/latentspace_specific/", max_points=2000, perplexity=100):
        dataset_name = "RotatedMNIST"
        if test_domains is None:
            test_domains = list(range(6))

        dom2col = LatentVisualizer._fixed_domain_colors_rotatedmnist()
        print("[RotatedMNIST domain color mapping]")
        for d in range(6):
            print(f"  domain {d}: {mcolors.to_hex(dom2col[d])}")

        domain_labels = ["0°", "15°", "30°", "45°", "60°", "75°"]

        LatentVisualizer._save_domain_legend_rotatedmnist(dom2col=dom2col, save_path=save_path, model_name=model_name, filename="RotatedMNIST_domain_legend.png", domain_labels=domain_labels)

        for td in test_domains:
            test_domain = [td]

            if model_name not in MODEL_DISPATCH:
                raise ValueError(f"Unknown model_name={model_name}. Not in MODEL_DISPATCH.")
            builder_name = MODEL_DISPATCH[model_name]

            cfg = ModelConfig(model_name)
            loader_config, model_config, train_config = cfg.resolve(dataset_name)
            model_config["model_name"] = model_name
            dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

            ModelLoaderClass = LatentVisualizer._get_class("methods.builder", builder_name)
            model_loader = ModelLoaderClass(dataloader, model_config, train_config)
            model_loader.model.to(model_loader.device)

            load_model(model_loader.model, model_loader.model_path)
            model_loader.model.eval()

            outputs = model_loader.encode(model_loader.train_loader)

            if not model_name.startswith("blender"):
                raise ValueError("This function currently supports blender-style (z_inv, z_spec) only.")

            enc_list, (_, _, d_list) = outputs

            z_spec_list = [enc_list[i][1] for i in range(len(enc_list))]
            z_spec = torch.cat(z_spec_list, dim=0)

            d = torch.cat(d_list, dim=0)
            d_np = d.reshape(-1).detach().cpu().numpy().astype(int)

            n = len(d_np)
            if n > max_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, max_points, replace=False)
                d_np = d_np[idx]
                z_spec = z_spec[idx]

            z = z_spec.detach().cpu().numpy()
            if z.shape[1] > 2:
                z2 = TSNE(n_components=2, perplexity=perplexity, random_state=get_global_seed()).fit_transform(z)
            else:
                z2 = z

            d_fix = d_np.copy()
            d_fix[d_fix >= td] += 1

            colors = np.array([dom2col[int(di)] for di in d_fix])

            fig = plt.figure(figsize=(5, 4.5), frameon=True)
            fig.suptitle(f"Test Domain: {domain_labels[td]}", fontsize=20, y=0.97)

            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 0.94])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.scatter(z2[:, 0], z2[:, 1], c=colors, s=10, alpha=0.85, linewidths=0)

            os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
            out = os.path.join(save_path, model_name, f"{dataset_name}_testdomain{td}_zspec_tsne.png")

            plt.savefig(out, bbox_inches="tight", pad_inches=0.1, dpi=600)
            plt.close(fig)
            print(f"[Saved] {out}")


if __name__ == '__main__':
    set_seed(42)

    # Plot latent space figures for all datasets and models
    data_list = ['RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 'TerraIncognita']
    model_list = ['blender']
    
    for dataset_name in data_list:
        num_domains = DATA_DISPATCH[dataset_name]
        test_domains = list(range(num_domains))

        for model_name, test_domain in product(model_list, test_domains):
            test_domain = [test_domain]
            LatentVisualizer.plot_latent(dataset_name, model_name, test_domain, save_path=f'./analyses/figures/latentspace/')

    # Plot for latent space figures (all elements) for RotatedMNIST -> Main Plot
    LatentVisualizer.save_all_latents_tsne_rotatedmnist(model_name="blender", test_domains=list(range(6)), save_path="./analyses/figures/latentspace_elements/",
                                                        max_points=2000, perplexity=100, domain_labels=["0°","15°","30°","45°","60°","75°"], class_labels=[str(i) for i in range(10)])

    # Plot for final specific latent space figures for RotatedMNIST -> Appendix Plot
    LatentVisualizer.save_specific_latent_tsne_rotatedmnist(model_name="blender", test_domains=list(range(6)), save_path="./analyses/figures/latentspace_specific/", max_points=2000, perplexity=100)