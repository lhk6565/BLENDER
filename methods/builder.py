import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.dataset import *
from methods.blender import Blender
from methods.diva import DIVA
from methods.dirt import DIRT
from methods.lfme import LFME
from methods.arith import *
from utils import *


class ModelLoaderBase:
    def __init__(self, dataloader, model_config, train_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = dataloader
        self.train_loader, self.val_loader, self.test_loader, (self.train_domain_dim, self.test_domain_dim) = self.dataloader.make_dataloader()
        self.model_config = model_config
        self.train_config = train_config

        # model
        self.model = self._build_model().to(self.device)

        # optimizer
        model_params = []
        backbone_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("embedding_net.resnet50"):
                backbone_params.append(p)
            else:
                model_params.append(p)  
        self.optimizer = torch.optim.Adam([{'params': model_params, 'lr': self.train_config['lr']},
                                           {'params': backbone_params, 'lr': 5e-5}])
        
        self.epochs = self.train_config['epochs']
        self.model_path = build_save_path(model_name=self.model_config['model_name'],
                                          dataset_name=self.dataloader.dataset_name,
                                          test_domain=str(self.dataloader.test_domain),
                                          seed=get_global_seed())
    
    def _early_stopping(self, curr_acc, epoch):
        """Early stopping based on validation accuracy."""
        if  curr_acc > self.best_acc:
            self.best_acc = curr_acc
            self.counter = 0
            save_model(self.model, self.model_path)
            tqdm.write(f'[Saved] best model at epoch {epoch+1} with best accuracy {self.best_acc:.4f}')
            return False
        else:
            self.counter += 1
            if self.counter >= self.train_config['patience']:
                tqdm.write(f'[Early stopping] at epoch {epoch+1}, best accuracy {self.best_acc:.4f}.')
                return True
        return False
    
    def _predict_y(self, x):
        raise NotImplementedError
    
    def _reconstruct(self, x):
        raise NotImplementedError
    
    def _encode(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_accuracy(self, loader):
        self.model.eval()
        correct, total = 0, 0

        for batch in loader:
            x, y, d = unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            d = d.to(self.device)

            pred_y = self._predict_y(x)
            correct += (pred_y == y).sum().item()
            total += y.size(0)

        return correct / max(total, 1)
    
    @torch.no_grad()
    def reconstruct(self, loader):
        self.model.eval()

        # first batch only
        for batch in loader:
            x, y, d = unpack_batch(batch)
            x = x.to(self.device)
            break
        x_rec, x_rec_comp = self._reconstruct(x)
        return x, x_rec, x_rec_comp
    
    @torch.no_grad()
    def encode(self, loader):
        self.model.eval()

        enc_list = []
        x_list, y_list, d_list = [], [], []

        for batch in loader:
            x, y, d = unpack_batch(batch)

            x = x.to(self.device)
            enc_out = self._encode(x)

            enc_list.append(enc_out)
            x_list.append(x.detach().cpu())
            y_list.append(y.detach().cpu())
            d_list.append(d.detach().cpu())

        return enc_list, (x_list, y_list, d_list)

    def train(self):
        self.best_acc = float('-inf')
        self.counter = 0

        for epoch in tqdm(range(self.epochs), desc='Epoch', leave=False):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            test_metrics = self.test_epoch(epoch)
            train_acc = self.evaluate_accuracy(self.train_loader)
            val_acc = self.evaluate_accuracy(self.val_loader)
            test_acc = self.evaluate_accuracy(self.test_loader)
            train_metrics['tr_accuracy'] = train_acc
            val_metrics['val_accuracy'] = val_acc
            test_metrics['te_accuracy'] = test_acc

            if self._early_stopping(val_metrics['val_accuracy'], epoch):
                break


class BlenderBuilder(ModelLoaderBase):
    def __init__(self, dataloader, model_config, train_config):
        super().__init__(dataloader, model_config, train_config)
    
    # -------------------------
    # model factory
    # -------------------------
    def _build_model(self):
        self.model_config['num_classes'] = self.dataloader.num_classes
        self.model_config['domain_dim'] = self.train_domain_dim

        return Blender(**self.model_config)
    
    def _schedule_control(self, epoch: int):
        beta_inv = min(epoch, 1000) / 1000
        beta_spec = min(epoch, 1000) / 1000
        lambda_hsic_dinv = min(epoch, 1000) / 1000
        lambda_hsic_inv_spec = min(epoch, 1000) / 1000
        
        return {'beta_inv': beta_inv, 'beta_spec': beta_spec, 'lambda_hsic_dinv': lambda_hsic_dinv, 'lambda_hsic_inv_spec': lambda_hsic_inv_spec}

    def _run_epoch(self, loader, epoch: int, prefix: str, train: bool):
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_n = 0
        # components: [recon, task, kld_inv, kld_spec, hsic_dinv, hsic_inv_spec]
        sum_components = torch.zeros(6, device=self.device)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc=f'{prefix} Batch', leave=False):
                x, y, d = unpack_batch(batch)
                x = x.to(self.device)
                y = y.to(self.device)
                d = F.one_hot(d.long(), num_classes=self.train_domain_dim).float().to(self.device)

                if train:
                    self.optimizer.zero_grad()

                _ = self.model(x)
                coef = self._schedule_control(epoch+1)
                loss, components = self.model.loss_function(x, y, d, **coef)
                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)

                sum_components += torch.stack([c.detach() for c in components]) * x.size(0)
                total_n += x.size(0)

        # avg per epoch
        avg_loss = total_loss / total_n
        avg_components = sum_components / total_n

        recon_loss, task_loss, kld_inv, kld_spec, hsic_dinv, hsic_inv_spec = avg_components.tolist()

        return {f'{prefix}_loss': avg_loss,
                f'{prefix}_recon_loss': recon_loss,
                f'{prefix}_task_loss': task_loss,
                f'{prefix}_kld_inv': kld_inv,
                f'{prefix}_kld_spec': kld_spec,
                f'{prefix}_hsic_dinv': hsic_dinv,
                f'{prefix}_hsic_inv_spec': hsic_inv_spec}

    def train_epoch(self, epoch: int):
        return self._run_epoch(self.train_loader, epoch, 'tr', train=True)

    def val_epoch(self, epoch: int):
        return self._run_epoch(self.val_loader, epoch, 'val', train=False)
    
    def test_epoch(self, epoch: int):
        return self._run_epoch(self.test_loader, epoch, 'te', train=False)
    
    @torch.no_grad()
    def _predict_y(self, x):
        x_rec, y_pred, (z_inv, z_spec) = self.model(x)
        return y_pred.argmax(dim=1)
    
    @torch.no_grad()
    def _reconstruct(self, x):
        x_rec, _, (z_inv, z_spec) = self.model(x)
        x_rec_inv = self.model.decode(z_inv, torch.zeros_like(z_spec))
        x_rec_spec = self.model.decode(torch.zeros_like(z_inv), z_spec)
        
        return x_rec.cpu(), (x_rec_inv.cpu(), x_rec_spec.cpu())
    
    @torch.no_grad()
    def _encode(self, x):
        z_inv, z_spec = self.model.encode(x)
        return (z_inv.cpu(), z_spec.cpu())


class DivaBuilder(ModelLoaderBase):
    def __init__(self, dataloader, model_config, train_config):
        super().__init__(dataloader, model_config, train_config)

    def _build_model(self):
        self.model_config['d_dim'] = self.train_domain_dim
        self.model_config['x_dim'] = int(np.prod(self.dataloader.input_shape))
        self.model_config['y_dim'] = self.dataloader.num_classes
        self.model_config['in_channels'] = self.dataloader.input_shape[0]
        return DIVA(self.model_config)
    
    def _schedule_beta(self, epoch: int):
        return min(epoch + 1, 100) / 100
    
    def _run_epoch(self, loader, epoch: int, prefix: str, train: bool):
        self.model.train() if train else self.model.eval()

        total_loss, total_class_y_loss = 0.0, 0.0
        total_n = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc=f'{prefix} Batch', leave=False):
                x, y, d = unpack_batch(batch)
                x = x.to(self.device)
                y = F.one_hot(y.long(), num_classes=self.dataloader.num_classes).float().to(self.device)
                d = F.one_hot(d.long(), num_classes=self.train_domain_dim).float().to(self.device)

                if train:
                    self.optimizer.zero_grad()
                    
                self.model.beta_d = self._schedule_beta(epoch)
                self.model.beta_x = self._schedule_beta(epoch)
                self.model.beta_y = self._schedule_beta(epoch)

                loss, class_y_loss = self.model.loss_function(d, x, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_class_y_loss += class_y_loss.item() * x.size(0)
                total_n += x.size(0)

        avg_loss = total_loss / total_n
        avg_class_y_loss = total_class_y_loss / total_n

        return {f'{prefix}_loss': avg_loss,
                f'{prefix}_task_loss': avg_class_y_loss,}
    
    def train_epoch(self, epoch: int):
        return self._run_epoch(self.train_loader, epoch, 'tr', train=True)
    
    def val_epoch(self, epoch: int):
        return self._run_epoch(self.val_loader, epoch, 'val', train=False)

    def test_epoch(self, epoch: int):
        return self._run_epoch(self.test_loader, epoch, 'te', train=False)
    
    @torch.no_grad()
    def _predict_y(self, x):
        y_pred = self.model.pred_y(x)
        return y_pred.argmax(dim=1)
    
    @torch.no_grad()
    def _reconstruct(self, x):
        x_rec, (x_rec_zd, x_rec_zx, x_rec_zy) = self.model.reconstruct(x)
        return x_rec.cpu(), (x_rec_zd.cpu(), x_rec_zx.cpu(), x_rec_zy.cpu())
    
    @torch.no_grad()
    def _encode(self, x):
        qzd, qzx, qzy = self.model.encode(x)
        return (qzd.cpu(), qzx.cpu(), qzy.cpu())


class DirtBuilder(ModelLoaderBase):
    def __init__(self, dataloader, model_config, train_config):
        super().__init__(dataloader, model_config, train_config)

    def _build_model(self):
        self.model_config['domain_dim'] = self.train_domain_dim
        self.model_config['image_shape'] = self.dataloader.input_shape
        self.model_config['test_domain'] = self.dataloader.test_domain
        self.model_config['num_classes'] = self.dataloader.num_classes

        self.ensure_starGAN()

        return DIRT(self.model_config)
    
    def ensure_starGAN(self):
        self.starGAN_path = build_save_path(model_name='stargan',
                                            dataset_name=self.dataloader.dataset_name,
                                            test_domain=str(self.dataloader.test_domain),
                                            seed=get_global_seed(),
                                            base_dir= f"./analyses/models/{self.model_config['model_name']}/")
        
        self.model_config['pretrained_stargan_path'] = self.starGAN_path

        if not os.path.isfile(self.model_config['pretrained_stargan_path']):
            from methods.dirt import StarGANTrainer
            stargan_trainer = StarGANTrainer((self.model_config, self.train_config))
            stargan_trainer.train(self.train_loader, self.val_loader)
        else:
            tqdm.write(f"Pretrained StarGAN loaded from {self.model_config['pretrained_stargan_path']}")

    def _run_epoch(self, loader, epoch: int, prefix: str, train: bool):
        self.model.train() if train else self.model.eval()

        total_loss = total_loss_cls = total_reg = total_acc = 0.0
        total_n = 0
        # components: [recon, task, kld_inv, kld_spec, ce_d, hsic_dinv, hsic_inv_spec]
        sum_components = torch.zeros(7, device=self.device)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc=f'{prefix} Batch', leave=False):
                x, y, d = unpack_batch(batch)
                x = x.to(self.device)
                y = y.to(self.device)
                d = d.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                loss, loss_cls, reg, acc = self.model.loss_function(x, y, d)

                if train:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item() * x.size(0)
                total_loss_cls += loss_cls.item() * x.size(0)
                total_reg += reg.item() * x.size(0)
                total_acc += acc.item() * x.size(0)
                total_n += x.size(0)

        avg_loss = total_loss / total_n
        avg_loss_cls = total_loss_cls / total_n
        avg_reg = total_reg / total_n
        avg_acc = total_acc / total_n

        return {f'{prefix}_loss': avg_loss,
                f'{prefix}_task_loss': avg_loss_cls,
                f'{prefix}_reg': avg_reg,
                f'{prefix}_acc': avg_acc}

    def train_epoch(self, epoch: int):
        return self._run_epoch(self.train_loader, epoch, 'tr', train=True)
    
    def val_epoch(self, epoch: int):
        return self._run_epoch(self.val_loader, epoch, 'val', train=False)

    def test_epoch(self, epoch: int):
        return self._run_epoch(self.test_loader, epoch, 'te', train=False)
    
    def _predict_y(self, x):
        logits, z = self.model(x)
        return logits.argmax(1)


class LFMEBuilder(ModelLoaderBase):
    def __init__(self, dataloader, model_config, train_config):
        super().__init__(dataloader, model_config, train_config)

        del self.optimizer
        self._build_optimizer()

    def _build_model(self):
        hparams = dict(self.model_config)

        return LFME(input_shape=self.dataloader.input_shape, num_classes=self.dataloader.num_classes, num_domains=self.train_domain_dim, hparams=hparams,)

    def _build_optimizer(self):
        """
        원본 LFME처럼 expert별 optimizer와 target optimizer를 분리한다.

        Expert를 하나씩 backward하므로 여러 ResNet의 activation graph를
        동시에 유지하지 않아 GPU 메모리도 절약된다.
        """
        lr = self.train_config["lr"]
        weight_decay = self.train_config.get("weight_decay", 0.0)

        self.expert_optimizers = [torch.optim.Adam(expert.parameters(), lr=lr, weight_decay=weight_decay,) for expert in self.model.experts]

        self.target_optimizer = torch.optim.Adam(self.model.target.parameters(), lr=lr, weight_decay=weight_decay,)

    def _split_by_domain(self, x, y, d):
        """
        하나의 mixed batch를 source domain별 minibatch로 분리한다.

        d가 0, ..., train_domain_dim - 1로 remap되어 있어야 한다.
        """
        d = d.long().view(-1)
        minibatches = []

        for domain_idx in range(self.train_domain_dim):
            mask = d == domain_idx

            if mask.any():
                minibatches.append((domain_idx, x[mask], y[mask]))

        if not minibatches:
            raise RuntimeError("현재 batch에 사용 가능한 source domain sample이 없습니다.")

        return minibatches

    def _run_epoch(self, loader, epoch, prefix, train):
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_task_loss = 0.0
        total_guidance_loss = 0.0
        total_expert_loss = 0.0
        total_n = 0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in tqdm(loader, desc=f"{prefix} Batch", leave=False,):
                x, y, d = unpack_batch(batch)

                x = x.to(self.device)
                y = y.long().to(self.device)
                d = d.long().to(self.device)

                # Validation/test에서는 inference model만 평가
                if not train:
                    target_logits = self.model(x)
                    task_loss = F.cross_entropy(target_logits, y)

                    batch_size = y.size(0)
                    total_loss += task_loss.item() * batch_size
                    total_task_loss += task_loss.item() * batch_size
                    total_n += batch_size
                    continue

                domain_minibatches = self._split_by_domain(x, y, d)

                all_x = []
                all_y = []
                expert_probabilities = []
                expert_losses = []

                # ---------------------------------------
                # 1. Source-domain experts 업데이트
                # ---------------------------------------
                for domain_idx, domain_x, domain_y in domain_minibatches:
                    
                    self.expert_optimizers[domain_idx].zero_grad()

                    expert_logits = self.model.forward_expert(domain_idx, domain_x,)

                    expert_loss = F.cross_entropy(expert_logits, domain_y,)

                    # Guidance는 expert update 이전의 output을 사용하고
                    # target loss가 expert에 역전파되지 않도록 detach
                    expert_probability = F.softmax(expert_logits.detach(), dim=1,)

                    expert_loss.backward()
                    self.expert_optimizers[domain_idx].step()

                    all_x.append(domain_x)
                    all_y.append(domain_y)
                    expert_probabilities.append(expert_probability)
                    expert_losses.append(expert_loss.detach())

                all_x = torch.cat(all_x, dim=0)
                all_y = torch.cat(all_y, dim=0)
                expert_probabilities = torch.cat(expert_probabilities, dim=0,)

                # ---------------------------------------
                # 2. Target model 업데이트
                # ---------------------------------------
                
                self.target_optimizer.zero_grad()

                target_logits = self.model(all_x)

                task_loss = F.cross_entropy(target_logits, all_y,)

                # 공식 구현과 동일하게:
                # target logits vs expert softmax probabilities
                guidance_loss = F.mse_loss(target_logits, expert_probabilities,)

                target_loss = (task_loss + self.train_config["lfe_reg"] * guidance_loss)

                target_loss.backward()
                self.target_optimizer.step()

                mean_expert_loss = torch.stack(expert_losses).mean()

                batch_size = all_y.size(0)

                # 원본 LFME의 반환 loss는 target loss에 해당
                total_loss += target_loss.item() * batch_size
                total_task_loss += task_loss.item() * batch_size
                total_guidance_loss += guidance_loss.item() * batch_size
                total_expert_loss += mean_expert_loss.item() * batch_size
                total_n += batch_size

        result = {f"{prefix}_loss": total_loss / max(total_n, 1),
                  f"{prefix}_task_loss": total_task_loss / max(total_n, 1),}

        if train:
            result.update({f"{prefix}_guidance_loss": total_guidance_loss / max(total_n, 1),
                           f"{prefix}_expert_loss": total_expert_loss / max(total_n, 1),})

        return result

    def train_epoch(self, epoch):
        return self._run_epoch(self.train_loader, epoch, "tr", train=True,)

    def val_epoch(self, epoch):
        return self._run_epoch(self.val_loader, epoch, "val", train=False,)

    def test_epoch(self, epoch):
        return self._run_epoch(self.test_loader, epoch, "te", train=False,)

    @torch.no_grad()
    def _predict_y(self, x):
        return self.model(x).argmax(dim=1)

    def _reconstruct(self, x):
        raise NotImplementedError("LFME는 reconstruction model이 아닙니다.")

    def _encode(self, x):
        raise NotImplementedError("LFME encode 분석은 별도로 구현해야 합니다.")


class ArithBuilder(ModelLoaderBase):
    """
    ARITH baseline using the paper's DomainBed protocol.

    Performance-critical design:
      - reads source-domain datasets directly from train_ds.datasets;
      - creates one persistent DataLoader per source domain;
      - avoids DomainBatchBank, Python per-image queues and sample.clone();
      - avoids per-parameter .item() synchronization in the training loop.
    """

    def __init__(self, dataloader, model_config, train_config):
        super().__init__(dataloader, model_config, train_config)

        # Replace ModelLoaderBase's generic optimizer with outer Adam.
        del self.optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config["lr"], weight_decay=self.train_config["weight_decay"])

        # Momentum-free inner SGD step size.
        self.inner_lr = self.model_config["inner_lr"]

        # Batch size for each source-domain inner step.
        self.domain_batch_size = int(self.model_config.get("domain_batch_size", self.train_config.get("batch_size", 32)))

        self.eval_step = int(self.train_config["eval_step"])
        self.log_step = int(self.train_config["log_step"])

        self.domain_weights = make_arithmetic_weights(self.train_domain_dim)

        # model_train.py calls set_seed(41), ..., set_seed(45).
        seed = get_global_seed()
        self.domain_rng = random.Random(seed)

        self.domain_train_loaders = self._build_domain_train_loaders(seed=seed)
        self.domain_train_iters = [iter(loader) for loader in self.domain_train_loaders]

    def _build_model(self):
        return ARITH(num_classes=self.dataloader.num_classes, backbone=self.model_config["backbone"],pretrained=self.model_config["pretrained"])

    def _build_domain_train_loaders(self, seed):
        """
        self.dataloader.train_ds is ConcatDomainDataset.
        Its `.datasets` field contains one training Subset per source domain.
        """

        train_ds = self.dataloader.train_ds
        domain_datasets = getattr(train_ds, "datasets", None)

        if domain_datasets is None:
            raise TypeError("ARITH requires self.dataloader.train_ds.datasets. "
                            "The current train dataset is not a ConcatDomainDataset.")

        domain_datasets = list(domain_datasets)

        if len(domain_datasets) != self.train_domain_dim:
            raise RuntimeError("Number of source-domain datasets does not match "
                               f"train_domain_dim: {len(domain_datasets)} "
                               f"vs {self.train_domain_dim}.")

        loader_cfg = self.dataloader.loader_cfg

        # This is workers per source-domain loader.
        num_workers = loader_cfg["domain_num_workers"]
        prefetch_factor = loader_cfg["domain_prefetch_factor"]

        domain_loaders = []

        for domain_index, domain_dataset in enumerate(domain_datasets):
            if len(domain_dataset) < self.domain_batch_size:
                raise ValueError(f"Source domain {domain_index} has only {len(domain_dataset)} training samples, which is smaller than domain_batch_size= {self.domain_batch_size}.")

            generator = torch.Generator()
            generator.manual_seed(int(seed) + domain_index)

            loader_kwargs = {"dataset": domain_dataset,
                             "batch_size": self.domain_batch_size,
                             "shuffle": True,
                             "drop_last": True,
                             "num_workers": num_workers,
                             "pin_memory": True,
                             "persistent_workers": num_workers > 0,
                             "generator": generator,}

            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = prefetch_factor

            domain_loaders.append(DataLoader(**loader_kwargs))

        return domain_loaders

    def _next_domain_batch(self, domain_index):
        try:
            batch = next(self.domain_train_iters[domain_index])
        except StopIteration:
            self.domain_train_iters[domain_index] = iter(self.domain_train_loaders[domain_index])
            batch = next(self.domain_train_iters[domain_index])

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError("A source-domain dataset must return at least (x, y).")

        return batch[0], batch[1]

    def train_epoch(self, iteration):
        """One complete ARITH outer iteration."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        clear_fast_weights(self.model)

        domain_order = list(range(self.train_domain_dim))
        self.domain_rng.shuffle(domain_order)

        fast_parameters = list(self.model.parameters())

        # Keep metrics on GPU and synchronize only once at return.
        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device, dtype=torch.long)
        total_n = 0

        for position, domain_index in enumerate(domain_order):
            x, y = self._next_domain_batch(domain_index)

            x = x.to(self.device, non_blocking=True)
            y = y.long().to(self.device, non_blocking=True)

            loss, logits, gradients = compute_domain_gradient(model=self.model, fast_parameters=fast_parameters, x=x, y=y,)

            # Exactly one momentum-free SGD inner step for this domain.
            fast_parameters, displacements = inner_sgd_step(model=self.model, gradients=gradients, inner_lr=self.inner_lr,)

            accumulate_arithmetic_gradients(model=self.model, displacements=displacements, weight=self.domain_weights[position],)

            batch_n = y.size(0)
            total_loss.add_(loss.detach() * batch_n)
            total_correct.add_((logits.argmax(dim=1) == y).sum())
            total_n += batch_n

        self.optimizer.step()
        clear_fast_weights(self.model)

        return {"tr_loss": (total_loss / max(total_n, 1)).item(),
                "tr_accuracy": (total_correct.float() / max(total_n, 1)).item()}

    def _run_eval(self, loader, prefix):
        self.model.eval()
        clear_fast_weights(self.model)

        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device, dtype=torch.long)
        total_n = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{prefix} Batch", leave=False):
                x, y, _ = unpack_batch(batch)

                x = x.to(self.device, non_blocking=True)
                y = y.long().to(self.device, non_blocking=True)

                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                batch_n = y.size(0)
                total_loss.add_(loss * batch_n)
                total_correct.add_((logits.argmax(dim=1) == y).sum())
                total_n += batch_n

        avg_loss = (total_loss / max(total_n, 1)).item()
        accuracy = (total_correct.float() / max(total_n, 1)).item()

        return {f"{prefix}_loss": avg_loss,
                f"{prefix}_task_loss": avg_loss,
                f"{prefix}_accuracy": accuracy}

    def val_epoch(self, iteration):
        return self._run_eval(self.val_loader, "val")

    def test_epoch(self, iteration):
        return self._run_eval(self.test_loader, "te")

    def train(self):
        """
        Run every outer iteration and choose the checkpoint with the highest
        combined source-validation accuracy. No early stopping is used.
        """

        self.best_acc = float("-inf")
        best_state = None
        last_iteration = -1

        progress = tqdm(range(self.epochs), desc="ARITH Iteration", leave=False)

        for iteration in progress:
            last_iteration = iteration
            train_metrics = self.train_epoch(iteration)

            should_eval = ((iteration + 1) % self.eval_step == 0 or iteration + 1 == self.epochs)

            if should_eval:
                val_metrics = self.val_epoch(iteration)
                val_accuracy = val_metrics["val_accuracy"]

                if val_accuracy > self.best_acc:
                    self.best_acc = val_accuracy
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

                    save_model(self.model, self.model_path)
                    tqdm.write(f"[Saved] ARITH best model at iteration {iteration + 1} with validation accuracy {self.best_acc:.4f}")

        # Evaluate the held-out target only after validation-based selection.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_metrics = self.test_epoch(last_iteration)

    @torch.no_grad()
    def _predict_y(self, x):
        clear_fast_weights(self.model)
        return self.model(x).argmax(dim=1)

    def _reconstruct(self, x):
        raise NotImplementedError("ARITH is a classification baseline.")

    def _encode(self, x):
        raise NotImplementedError("ARITH representation export is not implemented.")
    

if __name__ == '__main__':
    set_seed(42)

    dataset_name = 'RotatedMNIST'
    model_name = 'blender'
    test_domain = [1]

    cfg = ModelConfig(model_name)
    loader_config, model_config, train_config = cfg.resolve(dataset_name)
    model_config['model_name'] = model_name

    dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

    model_loader = BlenderBuilder(dataloader, model_config, train_config)

    model_loader.train()