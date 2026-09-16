"""MaPLe trained with Support-Resampled Meta Prompt Learning (SRMP).

SRMP changes only the organization of the few-shot support set and the
optimization procedure used during training. The MaPLe prompt architecture,
CLIP forward path, and original MaPLe classification loss are reused without
modification.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call

from dassl.data.data_manager import DataManager, build_data_loader
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY

from .maple import MaPLe
from .mmrl_SRMP import SkippingDatasetWrapper


class PairedSupportLoader:
    """Yield synchronized mini-batches from complementary support subsets."""

    def __init__(self, loader_a, loader_b):
        if len(loader_a) != len(loader_b):
            raise RuntimeError(
                "SRMP complementary loaders must contain the same number "
                "of mini-batches"
            )
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __len__(self):
        return len(self.loader_a)

    def __iter__(self):
        for batch_a, batch_b in zip(self.loader_a, self.loader_b):
            if batch_a is None or batch_b is None:
                print("SKIP_SRMP_BATCH reason=all_images_unreadable")
                continue
            yield batch_a, batch_b


@TRAINER_REGISTRY.register()
class MaPLe_SRMP(MaPLe):
    """Apply first-order, bidirectional SRMP optimization to MaPLe."""

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        ratio = cfg.TRAINER.MAPLE_SRMP.SPLIT_RATIO
        if not 0.0 < ratio < 1.0:
            raise ValueError("MAPLE_SRMP.SPLIT_RATIO must lie in (0, 1)")

    def build_data_loader(self):
        dm = DataManager(self.cfg, dataset_wrapper=SkippingDatasetWrapper)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    def build_model(self):
        # MaPLe owns the prompt architecture, loss, optimizer, and scheduler.
        super().build_model()
        self._full_train_loader = self.train_loader_x
        self._train_transform = build_transform(self.cfg, is_train=True)
        self._last_split_keys = None

    def _split_support_set(self, epoch):
        """Create per-class disjoint and complementary subsets A and B."""
        grouped = defaultdict(list)
        for datum in self.dm.dataset.train_x:
            grouped[int(datum.label)].append(datum)

        if not grouped:
            raise RuntimeError("SRMP received an empty training support set")

        class_sizes = {label: len(items) for label, items in grouped.items()}
        unique_sizes = set(class_sizes.values())
        if len(unique_sizes) != 1:
            raise RuntimeError(
                "SRMP requires the same K-shot support size for every class; "
                "got {}".format(class_sizes)
            )

        num_shots = unique_sizes.pop()
        split_ratio = self.cfg.TRAINER.MAPLE_SRMP.SPLIT_RATIO
        num_a = int(round(num_shots * split_ratio))
        num_b = num_shots - num_a
        if num_a <= 0 or num_b <= 0:
            raise RuntimeError(
                "SRMP requires two non-empty complementary subsets, but "
                "K={} and SPLIT_RATIO={} produce {}/{}".format(
                    num_shots, split_ratio, num_a, num_b
                )
            )
        if num_a != num_b:
            raise RuntimeError(
                "This paired mini-batch implementation requires an equal "
                "split; K={} and SPLIT_RATIO={} produce {}/{}".format(
                    num_shots, split_ratio, num_a, num_b
                )
            )

        # A non-negative base seed is deterministic. For cfg.SEED < 0, NumPy's
        # global RNG first supplies a fresh base seed for this training run.
        if self.cfg.SEED >= 0:
            epoch_seed = self.cfg.SEED + epoch
        else:
            epoch_seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.RandomState(epoch_seed)

        subset_a, subset_b = [], []
        split_keys = {}
        for label in sorted(grouped):
            items = grouped[label]
            order = rng.permutation(num_shots)
            samples_a = [items[index] for index in order[:num_a]]
            samples_b = [items[index] for index in order[num_a:]]

            keys_a = {
                (datum.impath, int(datum.label)) for datum in samples_a
            }
            keys_b = {
                (datum.impath, int(datum.label)) for datum in samples_b
            }
            original_keys = {
                (datum.impath, int(datum.label)) for datum in items
            }
            if len(keys_a) != num_a or len(keys_b) != num_b:
                raise RuntimeError(
                    "SRMP requires unique support samples within each class"
                )
            if keys_a & keys_b:
                raise RuntimeError("SRMP subsets A and B overlap")
            if keys_a | keys_b != original_keys:
                raise RuntimeError(
                    "SRMP subsets A and B are not complementary"
                )

            subset_a.extend(samples_a)
            subset_b.extend(samples_b)
            split_keys[label] = (keys_a, keys_b)

        return subset_a, subset_b, split_keys, num_a, num_b

    def before_epoch(self):
        subset_a, subset_b, split_keys, num_a, num_b = (
            self._split_support_set(self.epoch)
        )
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        # These loaders are created after CUDA initialization. Spawning new
        # workers here can deadlock, so only the dynamic SRMP loaders use the
        # main process.
        loader_cfg = self.cfg.clone()
        loader_cfg.defrost()
        loader_cfg.DATALOADER.NUM_WORKERS = 0
        loader_cfg.freeze()
        loader_kwargs = dict(
            cfg=loader_cfg,
            sampler_type="RandomSampler",
            batch_size=batch_size,
            tfm=self._train_transform,
            # Keep all samples even when a subset is not divisible by the
            # batch size. The transform itself is still the train transform.
            is_train=False,
        )
        loader_a = build_data_loader(
            data_source=subset_a,
            dataset_wrapper=SkippingDatasetWrapper,
            **loader_kwargs
        )
        loader_b = build_data_loader(
            data_source=subset_b,
            dataset_wrapper=SkippingDatasetWrapper,
            **loader_kwargs
        )
        self.train_loader_x = PairedSupportLoader(loader_a, loader_b)

        changed = (
            self._last_split_keys is not None
            and split_keys != self._last_split_keys
        )
        self._last_split_keys = split_keys
        print(
            "SRMP_SPLIT epoch={} classes={} subset_a={} subset_b={} "
            "per_class={}/{} changed_from_previous={}".format(
                self.epoch + 1,
                len(split_keys),
                len(subset_a),
                len(subset_b),
                num_a,
                num_b,
                changed,
            )
        )

    def _trainable_parameters(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

    def _inner_step(self, batch, inner_lr):
        """Return MaPLe loss and temporary first-order fast parameters."""
        image, label = self.parse_batch_train(batch)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        inner_loss = model(image, label)
        named_parameters = self._trainable_parameters()
        if not named_parameters:
            raise RuntimeError("SRMP found no trainable MaPLe parameters")

        gradients = torch.autograd.grad(
            inner_loss,
            [parameter for _, parameter in named_parameters],
            create_graph=False,
        )
        fast_parameters = {
            name: parameter - inner_lr * gradient.detach()
            for (name, parameter), gradient in zip(
                named_parameters, gradients
            )
        }
        return inner_loss.detach(), fast_parameters

    def _query_loss(self, fast_parameters, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # MaPLe's original training forward returns its own CE loss. SRMP does
        # not introduce or replace the base learner's loss.
        return functional_call(model, fast_parameters, (image, label))

    def _srmp_forward(self, batch_a, batch_b, inner_lr):
        # Both virtual adaptations start from the same current parameters.
        inner_a, fast_a = self._inner_step(batch_a, inner_lr)
        loss_a_to_b = self._query_loss(fast_a, batch_b)

        inner_b, fast_b = self._inner_step(batch_b, inner_lr)
        loss_b_to_a = self._query_loss(fast_b, batch_a)

        srmp_loss = 0.5 * (loss_a_to_b + loss_b_to_a)
        return inner_a, inner_b, loss_a_to_b, loss_b_to_a, srmp_loss

    def _get_inner_lr(self):
        inner_lr = self.cfg.TRAINER.MAPLE_SRMP.INNER_LR
        if inner_lr < 0:
            return self.get_current_lr()
        return inner_lr

    def forward_backward(self, batch):
        batch_a, batch_b = batch
        inner_lr = self._get_inner_lr()
        self.optim.zero_grad()

        if self.cfg.TRAINER.MAPLE.PREC == "amp":
            with autocast():
                result = self._srmp_forward(batch_a, batch_b, inner_lr)
            srmp_loss = result[-1]
            self.scaler.scale(srmp_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            result = self._srmp_forward(batch_a, batch_b, inner_lr)
            srmp_loss = result[-1]
            srmp_loss.backward()
            self.optim.step()

        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()

        inner_a, inner_b, loss_a_to_b, loss_b_to_a, _ = result
        return {
            "loss": srmp_loss.item(),
            "inner_a": inner_a.item(),
            "inner_b": inner_b.item(),
            "a_to_b": loss_a_to_b.detach().item(),
            "b_to_a": loss_b_to_a.detach().item(),
        }
