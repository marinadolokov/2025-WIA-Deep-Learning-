import torch
import numpy as np
import yaml
import random
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler


class BaseLearner:
    _adjectives = [
        'space','silent','fuzzy','crimson','ancient','clever','wild',
        'bold','gentle','mystic','electric','brilliant','sparse',
        'vivid','neon','lunar','solar','radiant','shy','stoic'
    ]
    _nouns = [
        'bear','falcon','tiger','phoenix','wolf','raven','lynx','otter',
        'serpent','horizon','comet','nebula','galaxy','rider','voyager',
        'pioneer','specter','shadow','monarch','oracle'
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler=None,
        metrics: dict=None,
        patience: int=10,
        model_dir: str=None,
        checkpoint_dir: str=None,
        use_random_name: bool=True,
        custom_name: str=None,
        model_name: str=None,
        verbose: bool=True,
        **train_params
    ):
        # Core setup
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.patience = patience
        self.verbose = verbose
        self.train_params = train_params

        # Run naming
        type_name = model_name or model.__class__.__name__
        if custom_name:
            base = custom_name
        elif use_random_name:
            base = f"{random.choice(self._adjectives)}_{random.choice(self._nouns)}"
        else:
            base = 'run'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{type_name}_{base}_{timestamp}"

        # Directories
        root = Path(model_dir) if model_dir else Path.cwd()
        self.run_dir = root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_dir:
            chk_root = Path(checkpoint_dir)
            self.checkpoint_dir = chk_root / self.run_name
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = self.run_dir

        # Paths
        self.model_path = self.run_dir / f"{self.run_name}.pt"
        self.checkpoint_path = self.checkpoint_dir / f"{self.run_name}.ckpt.pt"
        self.log_path = self.run_dir / f"{self.run_name}.log"
        self.config_path = self.run_dir / f"{self.run_name}.yaml"

        # Save config
        crit_params = {}
        if hasattr(criterion, 'state_dict'):
            for k,v in criterion.state_dict().items():
                if isinstance(v, torch.Tensor) and v.numel()==1:
                    crit_params[k] = v.item()
        cfg = {
            'model_type': type_name,
            'optimizer': optimizer.__class__.__name__,
            'criterion': criterion.__class__.__name__,
            'criterion_params': crit_params,
            'scheduler': scheduler.__class__.__name__ if scheduler else None,
            'metrics': list(self.metrics.keys()),
            'patience': patience,
            **train_params
        }
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(cfg, f)

        # Device & history
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': []}
        for name in self.metrics:
            self.history[f'val_{name}'] = []
        self.best_score = -np.inf
        self.epochs_no_improve = 0

    def _to_device(self, x):
        if isinstance(x, (list,tuple)):
            return [xx.to(self.device) for xx in x]
        return x.to(self.device)

    def _forward(self, x):
        if isinstance(x, (list,tuple)):
            return self.model(*x)
        else:
            return self.model(x)

    def _prepare_targets(self, y):
        raise NotImplementedError

    def _raw_targets(self, y):
        raise NotImplementedError

    def _get_scores(self, logits):
        raise NotImplementedError

    def _get_preds(self, scores):
        raise NotImplementedError

    def _print_metrics(self, metrics: dict, prefix: str, wrap_n: int = 3):
        # Print header
        print(prefix)
        max_key = max(len(k) for k in metrics)
        items = [f"{k.ljust(max_key)}: {v:7.4f}" for k,v in metrics.items()]
        indent = ' ' * (len(prefix)+3)
        for i in range(0, len(items), wrap_n):
            print(indent + ' | '.join(items[i:i+wrap_n]))

    def _build_checkpoint_dict(self, epoch: int):
        ckpt = {'epoch': epoch,
                'state': self.model.state_dict(),
                'opt': self.optimizer.state_dict()}
        if hasattr(self.criterion, 'state_dict'):
            cs = self.criterion.state_dict()
            if cs: ckpt['crit'] = cs
        if self.scheduler and hasattr(self.scheduler, 'state_dict'):
            ckpt['sched'] = self.scheduler.state_dict()
        return ckpt

    def train_one_epoch(self):
        self.model.train()
        total = 0.0
        for x,y in self.train_loader:
            x = self._to_device(x)
            y_t = self._prepare_targets(y)
            self.optimizer.zero_grad()
            logits = self._forward(x)
            loss = self.criterion(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * y_t.size(0)
        return total / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total = 0.0
        preds, scores, trues = [], [], []
        with torch.no_grad():
            for x,y in self.val_loader:
                x = self._to_device(x)
                y_t = self._prepare_targets(y)
                logits = self._forward(x)
                loss = self.criterion(logits, y_t)
                total += loss.item() * y_t.size(0)
                sc = self._get_scores(logits)
                pr = self._get_preds(sc)
                preds.append(pr); scores.append(sc); trues.append(self._raw_targets(y))
        y_pred = np.concatenate(preds, axis=0)
        y_score = np.concatenate(scores, axis=0)
        y_true  = np.concatenate(trues, axis=0)
        avg_loss = total / len(self.val_loader.dataset)
        results = {n: fn(y_true, y_pred, y_score) for n,fn in self.metrics.items()}
        return avg_loss, results

    def fit(self, epochs: int):
    # print run header
        print(f"### Starting {self.run_name}")
        params = dict(
            model=self.model.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
            criterion=self.criterion.__class__.__name__,
            scheduler=self.scheduler.__class__.__name__ if self.scheduler else None,
            patience=self.patience,
            metrics=list(self.metrics.keys()),
            **self.train_params
        )
        for k, v in params.items():
            print(f"  {k}: {v}")

        # open log
        log_file = open(self.log_path, 'w')
        header = 'epoch,train_loss,val_loss,' + ','.join(self.metrics.keys()) + '\n'
        log_file.write(header)
        start_epoch = 0
        best_val_loss = float('inf')
        try:
            for ep in range(1, epochs+1):
                start_epoch = ep
                tr = self.train_one_epoch()
                vl, met = self.validate()

                # record history
                self.history['train_loss'].append(tr)
                self.history['val_loss'].append(vl)
                for k, v in met.items():
                    self.history[f'val_{k}'].append(v)

                # log
                log_file.write(
                    f"{ep},{tr:.4f},{vl:.4f}," + ",".join(f"{met[k]:.4f}" for k in met) + "\n"
                )

                # verbose output
                if self.verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    prefix = f"Epoch {ep:03d} — loss {tr:.4f}/{vl:.4f} — lr {lr:.1e}"
                    self._print_metrics(met, prefix)

                # save checkpoint
                torch.save(self._build_checkpoint_dict(ep), str(self.checkpoint_path))

                # best-model and early stopping
                if vl < best_val_loss - 1e-4:
                    best_val_loss = vl
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), str(self.model_path))
                else:
                    self.epochs_no_improve += 1

                if self.scheduler:
                    self.scheduler.step(vl)

                if self.epochs_no_improve >= self.patience:
                    print("Early stopping.")
                    break
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {start_epoch}. Saving checkpoint to '{self.checkpoint_path}'.")
            torch.save(self._build_checkpoint_dict(start_epoch), str(self.checkpoint_path))
        finally:
            log_file.close()
        return self.history


class MultiLabelLearner(BaseLearner):
    def __init__(self, *args, threshold: float=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _prepare_targets(self, y):
        return y.to(self.device).float()

    def _raw_targets(self, y):
        return y.cpu().numpy().astype(int)

    def _get_scores(self, logits):
        return torch.sigmoid(logits).cpu().numpy()

    def _get_preds(self, scores):
        return (scores > self.threshold).astype(int)


class MultiClassLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_targets(self, y):
        return y.to(self.device).long()

    def _raw_targets(self, y):
        return y.cpu().numpy()

    def _get_scores(self, logits):
        return torch.softmax(logits, dim=1).cpu().numpy()

    def _get_preds(self, scores):
        return np.argmax(scores, axis=1)




def compute_debiasing_sampler(
    dbvae,        # your trained (or jointly trained) DB-VAE
    dataset,      # real-task dataset, returns (x, y)
    batch_size=128,
    device='cpu',
    n_bins=30,
    eps=1e-6
):
    """
    Runs all x in `dataset` through dbvae.encode → mu,
    builds per-dimension histograms of mu, then
    weights each sample by the mean of 1/freq across dims.
    Returns a WeightedRandomSampler.
    """
    dbvae.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mus = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, _ = dbvae.encode(x)
            all_mus.append(mu.cpu())
    mus = torch.cat(all_mus, dim=0)       # shape: (N, latent_dim)
    N, D = mus.shape

    # inverse-frequencies for each dimension
    inv_freqs = torch.zeros(D, N)
    for d in range(D):
        vals = mus[:, d]
        hist = torch.histc(vals, bins=n_bins, min=vals.min(), max=vals.max())
        edges = torch.linspace(vals.min(), vals.max(), n_bins+1)
        bin_idx = torch.bucketize(vals, edges) - 1
        freq = hist[bin_idx]
        inv_freqs[d] = 1.0 / (freq + eps)

    # sample weight = average inverse-frequency across dims
    weights = inv_freqs.mean(dim=0)      # (N,)
    # normalize so sum(weights)=N
    weights = weights / weights.sum() * N
    # ensure no zeros
    weights = weights.clamp(min=eps)

    return WeightedRandomSampler(weights, num_samples=N, replacement=True)



def train_dbvae(model, dataset, epochs=20, batch_size=128, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = DBVAELoss()
    sampler   = None
    for epoch in range(1, epochs+1):
        # Refresh sampler every 5 epochs
        if epoch % 5 == 1:
            sampler = compute_debiasing_sampler(model, dataset, batch_size, device)
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler,
                            shuffle=(sampler is None))
        model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, logits = model(x)
            loss = loss_func(recon, x, mu, logvar, logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch} avg loss: {total_loss/len(dataset):.4f}")
        
        
