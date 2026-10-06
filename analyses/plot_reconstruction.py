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

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class ReconVisualizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_class(mod_name: str, cls_name: str):
        module = importlib.import_module(mod_name)
        return getattr(module, cls_name)
    
    @staticmethod
    def _standardize_reconstruction(model_name, x, x_rec, x_rec_comp):
        if model_name.startswith("blender"):
            x_rec_inv, x_rec_spec = x_rec_comp
            data_rows = [x, x_rec, x_rec_inv, x_rec_spec]
            row_labels = ["Original", "Reconstruction", "Invariant", "Specific"]

        elif model_name == "diva":
            x_rec_zd, x_rec_zx, x_rec_zy = x_rec_comp
            data_rows = [x, x_rec, x_rec_zd, x_rec_zx, x_rec_zy]
            row_labels = ["Original", "Reconstruction", "z_d", "z_x", "z_y"]

        return data_rows, row_labels
        
    @staticmethod
    def _to_image(x_rec_comp, top_N=1):
        if isinstance(x_rec_comp, (list, tuple)):
            return type(x_rec_comp)(ReconVisualizer._to_image(t) for t in x_rec_comp)

        if x_rec_comp.dim() == 4:
            # for 1 channel output
            if x_rec_comp.size(1) == 1:
                # prob thresholding
                threshold = 0.45
                x_rec_comp = (x_rec_comp > threshold).float()

                return x_rec_comp
            
            # 2. for 256 channel output
            if x_rec_comp.size(1) == 256:
                k = min(int(top_N), x_rec_comp.size(1))

                # weighted average (optional)
                top_val, top_idx = torch.topk(x_rec_comp, k=k, dim=1, largest=True)
                w = torch.softmax(top_val, dim=1)
                rec = (top_idx.float() * w).sum(dim=1, keepdim=True) / 255.0
                return rec
                
        return x_rec_comp
    
    # initial code (Not used for final paper, but can be used for quick checks)
    @staticmethod
    def plot_reconstruction(dataset_name, model_name, test_domain=[5], save_path='analyses/figures/reconstruction/'):        
        if model_name in MODEL_DISPATCH:
            builder_name = MODEL_DISPATCH[model_name]

            cfg = ModelConfig(model_name)
            loader_config, model_config, train_config = cfg.resolve(dataset_name)
            model_config['model_name'] = model_name
            dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

            ModelLoaderClass = ReconVisualizer._get_class('methods.builder', builder_name)
            model_loader = ModelLoaderClass(dataloader, model_config, train_config)
            model_loader.model.to(model_loader.device)

            load_model(model_loader.model, model_loader.model_path)
            model_loader.model.eval()

            x, x_rec, x_rec_comp = model_loader.reconstruct(model_loader.test_loader)
            x_rec = ReconVisualizer._to_image(x_rec)
            x_rec_comp = ReconVisualizer._to_image(x_rec_comp)

            data_rows, row_labels = ReconVisualizer._standardize_reconstruction(model_name, x, x_rec, x_rec_comp)

        nrows = len(data_rows)
        ncols = min(8, data_rows[0].shape[0])

        np_rows = [row_tensor[:ncols].cpu().numpy() for row_tensor in data_rows]

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows), gridspec_kw={'wspace': 0, 'hspace': 0})

        for r, (images, label) in enumerate(zip(np_rows, row_labels)):
            for c in range(ncols):
                ax = axes[r, c]
                ax.imshow(images[c].squeeze(), cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 0].text(-0.1, 0.5, label, va='center', ha='right',
            transform=axes[r, 0].transAxes, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.suptitle(f'Reconstruction Comparison of {model_name} (Test Domain: {test_domain[0]})', fontsize=16)

        if save_path:
            os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
            save_path = os.path.join(save_path, model_name, f'{dataset_name}_{test_domain}.pdf')
            plt.savefig(save_path)
            print(f'Reconstruction figure saved to {save_path}')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    # main reconstruction
    @staticmethod
    def plot_reconstruction_6X1(save_path="analyses/figures/recon_grids_6x1/", models=("blender", "diva"), num_samples=100, dpi=600):
        dataset_name = "RotatedMNIST"
        domains=(0,1,2,3,4,5)
        save_path = Path(save_path)

        def _save_grid(images_6, out_path):
            n = len(images_6)
            fig = plt.figure(figsize=(2.2, 2.2*n), frameon=False)
            fig.patch.set_facecolor("black")

            for r, img in enumerate(images_6):
                ax = fig.add_axes([0, 1-(r+1)/n, 1, 1/n])
                ax.set_axis_off()

                if torch.is_tensor(img):
                    img = img.detach().cpu().numpy()

                img = np.squeeze(img)
                ax.imshow(img, cmap="gray")

            fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
            plt.close(fig)

        # load test dataset per domain
        cfg0 = ModelConfig(models[0])
        loader_cfg0, _, _ = cfg0.resolve(dataset_name)
        loader_cfg0 = dict(loader_cfg0)
        loader_cfg0["num_workers"] = 0

        domain_test_datasets = {}
        for td in domains:
            dataloader = ImageDataLoader(dataset_name, test_domain=[td], loader_cfg=loader_cfg0)
            _, _, test_loader, _ = dataloader.make_dataloader()
            domain_test_datasets[td] = test_loader.dataset

        # minimum length across domains
        min_len = min(len(domain_test_datasets[td]) for td in domains)
        num_samples = min(num_samples, min_len)
        print(f"Saving {num_samples} samples per domain")

        # per-sample loop (index k is global across domains)
        for k in range(num_samples):

            fixed = {}
            for td in domains:
                ds = domain_test_datasets[td]
                x, y = ds[k]
                x = x.unsqueeze(0)
                y = torch.tensor([y])
                fixed[td] = (x, y)

            # save original image
            original_imgs = []
            for td in domains:
                x_sel, _ = fixed[td]
                x_img = ReconVisualizer._to_image(x_sel)
                original_imgs.append(x_img[0])

            out_dir = save_path / "original"
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_grid(original_imgs, out_dir / f"{k:03d}.png")

            # model reconstructions
            for model_name in models:

                builder_name = MODEL_DISPATCH[model_name]
                if model_name.startswith("blender"):
                    keys = ["reconstruction", "z_inv", "z_spec"]
                else:
                    keys = ["reconstruction", "z_d", "z_x", "z_y"]

                collected = {kk: [] for kk in keys}

                for td in domains:
                    cfg = ModelConfig(model_name)
                    loader_config, model_config, train_config = cfg.resolve(dataset_name)
                    loader_config = dict(loader_config)
                    model_config = dict(model_config)
                    model_config["model_name"] = model_name
                    loader_config["num_workers"] = 0

                    dataloader = ImageDataLoader(dataset_name, test_domain=[td], loader_cfg=loader_config)

                    ModelLoaderClass = ReconVisualizer._get_class("methods.builder", builder_name)
                    model_loader = ModelLoaderClass(dataloader, model_config, train_config)
                    model_loader.model.to(model_loader.device)

                    load_model(model_loader.model, model_loader.model_path)
                    model_loader.model.eval()

                    # fixed sample loader
                    x_sel, y_sel = fixed[td]
                    fixed_loader = DataLoader(TensorDataset(x_sel, y_sel), batch_size=1, shuffle=False, num_workers=0)

                    x, x_rec, x_rec_comp = model_loader.reconstruct(fixed_loader)

                    x_rec = ReconVisualizer._to_image(x_rec)
                    x_rec_comp = ReconVisualizer._to_image(x_rec_comp)

                    if model_name.startswith("blender"):
                        inv, spec = x_rec_comp
                        collected["reconstruction"].append(x_rec[0])
                        collected["z_inv"].append(inv[0])
                        collected["z_spec"].append(spec[0])
                    else:
                        zd, zx, zy = x_rec_comp
                        collected["reconstruction"].append(x_rec[0])
                        collected["z_d"].append(zd[0])
                        collected["z_x"].append(zx[0])
                        collected["z_y"].append(zy[0])

                # save
                for kk in keys:
                    out_dir = save_path / model_name / kk
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _save_grid(collected[kk], out_dir / f"{k:03d}.png")

            print(f"[Saved sample {k:03d}]")

    # sampling based reconstruction
    @staticmethod
    def plot_sampled_reconstruction(model_name="blender", split="train", mode="eval", num_sets=20, save_path="analyses/figures/recon_sampling/",
                                    max_batches_collect=100_000, dpi=600, force_sample_spec=True, seed=0):

        dataset_name = "RotatedMNIST"
        base_domain=0
        test_domain = 5

        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        if mode not in ["eval", "train"]:
            raise ValueError("mode must be 'eval' or 'train'")
        if model_name not in MODEL_DISPATCH:
            raise ValueError(f"Unknown model_name={model_name}. Not in MODEL_DISPATCH.")

        # seed
        np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # load model
        builder_name = MODEL_DISPATCH[model_name]
        cfg = ModelConfig(model_name)
        loader_cfg_base, model_cfg, train_cfg = cfg.resolve(dataset_name)
        loader_cfg_base = dict(loader_cfg_base)
        loader_cfg_base["num_workers"] = 0

        model_cfg = dict(model_cfg)
        model_cfg["model_name"] = model_name

        # tmp dataloader for model loading (no matter which domain)
        tmp_td = [test_domain] if test_domain is not None else [0]
        dataloader_tmp = ImageDataLoader(dataset_name, test_domain=tmp_td, loader_cfg=loader_cfg_base)

        ModelLoaderClass = ReconVisualizer._get_class("methods.builder", builder_name)
        model_loader = ModelLoaderClass(dataloader_tmp, model_cfg, train_cfg)
        model_loader.model.to(model_loader.device)

        load_model(model_loader.model, model_loader.model_path)

        if mode == "eval":
            model_loader.model.eval()
        else:
            model_loader.model.train()

        device = model_loader.device
        model = model_loader.model

        # reparameterization function (for sampling)
        def _reparam(mu, logvar):
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps

        # domain mapping
        all_domains = list(range(6))  # RotatedMNIST fixed
        if test_domain is None:
            train_domains = all_domains[:]  # 6
        else:
            test_domain = int(test_domain)
            train_domains = [d for d in all_domains if d != test_domain]  # 5

        if base_domain not in train_domains:
            raise ValueError(f"base_domain={base_domain} is not in train_domains={train_domains}. "
                             f"(If LODO and test_domain==base_domain, base_domain prior/encode cannot be formed.)")
        
        domain_to_idx = {dom: i for i, dom in enumerate(train_domains)}

        def _one_hot_domain(domain_id, n):
            d = torch.zeros(n, model.domain_dim, device=device)
            d[:, domain_to_idx[int(domain_id)]] = 1.0
            return d

        # collect reference image sets from base_domain
        # result: x_sets[s] shape [10,C,H,W]
        dataloader_base = ImageDataLoader(dataset_name, test_domain=[base_domain], loader_cfg=loader_cfg_base)
        train_loader, _, test_loader, _ = dataloader_base.make_dataloader()
        use_loader = train_loader if split == "train" else test_loader

        # num sets per digit class
        buckets = {c: [] for c in range(10)}  # c -> list[tensor(C,H,W)]
        done = False

        for bi, batch in enumerate(use_loader):
            if bi >= max_batches_collect:
                break

            x = batch[0]
            y = batch[1]
            y_np = y.reshape(-1).detach().cpu().numpy().astype(int)

            for i in range(x.size(0)):
                c = int(y_np[i])
                if 0 <= c < 10 and len(buckets[c]) < num_sets:
                    buckets[c].append(x[i].detach().cpu())
                if all(len(buckets[k]) >= num_sets for k in range(10)):
                    done = True
                    break

            if done:
                break

        missing = {c: num_sets - len(buckets[c]) for c in range(10) if len(buckets[c]) < num_sets}
        if missing:
            raise RuntimeError(f"Not enough samples in base_domain={base_domain}. Missing counts: {missing}")

        # x_sets: list length num_sets, each is [10,C,H,W]
        x_sets = []
        for s in range(num_sets):
            x_ref = torch.stack([buckets[c][s] for c in range(10)], dim=0).to(device)
            x_sets.append(x_ref)

        # save loop
        save_path = Path(save_path) / model_name / f"base{base_domain}_split{split}_mode{mode}" / (f"lodo_td{test_domain}" if test_domain is not None else "all_domains")
        save_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for s, x_ref in enumerate(x_sets):
                out_dir = save_path
                out_dir.mkdir(parents=True, exist_ok=True)
                out_pdf = out_dir / f"set_{s:03d}.png"

                # 1) base_domain image -> z_inv
                z_inv, _ = model.encode(x_ref)  # [10,z_inv_dim]

                # 2) generate images for each domain (rows x cols)
                rows = len(train_domains)
                cols = 10

                gen_imgs = []
                for d in train_domains:
                    d_oh = _one_hot_domain(d, n=10)
                    pred_mu, pred_logvar = model.pred_spec_dist(d_oh)

                    if (mode == "train") or force_sample_spec:
                        z_spec = _reparam(pred_mu, pred_logvar)
                    else:
                        z_spec = pred_mu

                    x_hat = model.decode(z_inv, z_spec)
                    x_hat = ReconVisualizer._to_image(x_hat)
                    x_np = x_hat.detach().cpu().numpy()  # [10,1,H,W] or [10,H,W]
                    gen_imgs.append(x_np)

                cell_w = 1.0
                cell_h = 1.0
                fig_w = cols * cell_w
                fig_h = rows * cell_h

                fig = plt.figure(figsize=(fig_w, fig_h), frameon=False)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

                for r in range(rows):
                    for c in range(cols):
                        left = c / cols
                        bottom = 1.0 - (r + 1) / rows
                        width = 1.0 / cols
                        height = 1.0 / rows

                        ax = plt.Axes(fig, [left, bottom, width, height])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        img = gen_imgs[r][c]
                        img = np.squeeze(img)
                        ax.imshow(img, cmap="gray")

                fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0, dpi=dpi)
                plt.close(fig)

                print(f"[Saved PDF] {out_pdf}")



if __name__ == '__main__':
    set_seed(42)

    # Plot reconstruction figures for all datasets and models
    data_list = ['RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 'TerraIncognita']
    model_list = ['blender']
    
    for dataset_name in data_list:
        num_domains = DATA_DISPATCH[dataset_name]
        test_domains = list(range(num_domains))

        for model_name, test_domain in product(model_list, test_domains):
            test_domain = [test_domain]
            if dataset_name == 'RotatedMNIST':
                ReconVisualizer.plot_reconstruction(dataset_name, model_name, test_domain, save_path=f'./analyses/figures/reconstruction/')

    # Plot for final reconstruction grids (6x1) for RotatedMNIST
    ReconVisualizer.plot_reconstruction_6X1(save_path="./analyses/figures/final/recon_grids_6x1/", num_samples=100)

    ReconVisualizer.plot_sampled_reconstruction(model_name="blender", num_sets=50, save_path="./analyses/figures/final/recon_sampling/", force_sample_spec=True, dpi=1200)