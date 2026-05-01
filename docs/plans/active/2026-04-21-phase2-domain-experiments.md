# GPTGAT2 Phase 2: Domain Integration & Experiment Validation

> **Status:** Phase 1 complete (Tasks 1-5 in `2026-03-15-model-p-pp-training-implementation.md`).
> This plan picks up from where Phase 1 left off.

**Goal:** Close the multi-dataset training loop. Prove whether PP brings benefit, wire domain adaptation as a first-class loss, and produce a reproducible experiment record for the current project.

**Current baseline state:**
- P-only: trains and produces metrics ✓
- P+PP: starts and runs but semantic validation incomplete
- Domain adaptation: `domain_adaptation.py` exists but is never imported in `train.py`
- No formal experiment comparison yet

---

## Task 1: Wire DomainAdaptationModule into train.py

**Why this is Task 1:** The domain loss is the core contribution beyond plain P+PP mixing. Everything else in Task 2 only makes sense if we can confirm domain alignment is actually happening.

**Files:**
- Modify: `E:/GPTGAT2/code/train.py`
- Modify: `E:/GPTGAT2/code/config.yaml` (and config.p_pp_full.yaml)
- Test: `E:/GPTGAT2/code/tests/test_p_pp_domain_and_forgetting.py` (create)

### Step 1: Write the failing test

```python
# tests/test_p_pp_domain_and_forgetting.py
import unittest
import torch

class TestDomainAdaptationIntegration(unittest.TestCase):

    def test_domain_adaptation_module_importable_from_train(self):
        """DomainAdaptationModule must be importable via the training module."""
        from domain_adaptation import DomainAdaptationModule
        da = DomainAdaptationModule(feature_dim=32, num_domains=3)
        feats = torch.randn(8, 32)
        labels = torch.randint(0, 3, (8,))
        out = da(feats, labels)
        self.assertIn('total', out)
        self.assertTrue(out['total'].requires_grad or out['total'].item() >= 0)

    def test_train_step_logs_domain_loss(self):
        """A training step with domain_weight > 0 must produce a nonzero domain_loss key."""
        # This test verifies the stats dict returned by train_one_epoch includes domain_loss
        # Will FAIL until train.py integrates domain adaptation
        from train import build_p_pp_loaders_from_config, get_training_stage
        # At minimum, domain_weight config key must be recognized by stage_loss_weights
        from train import stage_loss_weights
        weights = stage_loss_weights('B', {'stage_B_recon_weight': 1.0,
                                           'stage_B_cls_weight': 1.0,
                                           'stage_B_domain_weight': 0.5})
        self.assertIn('domain_weight', weights)
        self.assertAlmostEqual(weights['domain_weight'], 0.5)
```

Run: `python -m unittest tests.test_p_pp_domain_and_forgetting`
Expected: FAIL (domain_weight key not in stage_loss_weights, domain_loss not logged)

### Step 2: Implementation

**2a. Add domain_weight to stage_loss_weights in train.py**

In `stage_loss_weights(stage, config)`, add:
```python
'domain_weight': float(config.get(f'stage_{stage}_domain_weight',
                       config.get('domain_weight', 0.0)))
```

**2b. Import and instantiate DomainAdaptationModule in main()**

After model construction, add:
```python
from domain_adaptation import DomainAdaptationModule, get_domain_id_mapping, get_lambda_schedule

_domain_weight = float(config.get('domain_weight', 0.0))
_use_domain = _domain_weight > 0.0
domain_module = None
if _use_domain:
    _feature_dim = getattr(model, 'latent_dim', config.get('latent_dim', 64))
    _num_domains = int(config.get('num_domain_classes', 8))  # P+PP total datasets
    domain_module = DomainAdaptationModule(
        feature_dim=_feature_dim,
        num_domains=_num_domains,
        use_adversarial=bool(config.get('domain_use_adversarial', True)),
        use_mmd=bool(config.get('domain_use_mmd', True)),
        use_coral=bool(config.get('domain_use_coral', False)),
        adversarial_weight=float(config.get('domain_adversarial_weight', 1.0)),
        mmd_weight=float(config.get('domain_mmd_weight', 0.1)),
    ).to(device)
```

**2c. Pass domain_module to train_one_epoch**

In `train_one_epoch` signature, add:
```python
domain_module=None,
domain_weight: float = 0.0,
```

Inside the batch loop, after the model forward pass (which already returns latent):
```python
if domain_module is not None and domain_weight > 0.0:
    # latent shape: (T, C) or (B, D) — pool if needed
    latent_pooled = latent.mean(dim=0) if latent.ndim > 1 else latent
    # sources are the dataset names in the batch (from pp_metadata or source field)
    # domain_labels come from batch source slot (index 3 in collate output)
    domain_ids = _sources_to_domain_ids(sources_in_batch, domain_id_map)
    if domain_ids is not None:
        da_losses = domain_module(latent_pooled.unsqueeze(0).expand(len(domain_ids), -1),
                                  domain_ids)
        loss = loss + domain_weight * da_losses['total']
        domain_loss_accum += da_losses['total'].item()
```

**2d. GRL lambda scheduling in epoch loop**

In the epoch loop, before `train_one_epoch`, add:
```python
if domain_module is not None:
    lam = get_lambda_schedule(epoch, max_epochs,
                              schedule_type=config.get('domain_lambda_schedule', 'exp'))
    domain_module.set_lambda(lam)
```

**2e. Log domain_loss in TensorBoard and epoch stats**

**2f. Add config keys to config.yaml and config.p_pp_full.yaml**

```yaml
# Domain adaptation
domain_weight: 0.1
domain_use_adversarial: true
domain_use_mmd: true
domain_use_coral: false
domain_adversarial_weight: 1.0
domain_mmd_weight: 0.1
domain_lambda_schedule: exp   # 'linear', 'exp', 'step'
num_domain_classes: 8         # total distinct dataset sources
```

### Step 3: Run test to verify passes

Run: `python -m unittest tests.test_p_pp_domain_and_forgetting`
Expected: PASS

### Step 4: Smoke check domain loss appears in logs

```bash
python code/train.py --config code/config.smoke.yaml
```
Verify TensorBoard shows `train/domain_loss` and it is nonzero by epoch 2.

---

## Task 2: Diagnose and validate P+PP training semantics

**Why:** The summary (§十一) flagged that train-side metrics were long-term zero during a P+PP run. We need to confirm PP batches actually produce gradients and that P batches still drive classification improvement.

**Files:**
- Create: `E:/GPTGAT2/code/tests/test_p_pp_semantic_validation.py`
- Read: `E:/GPTGAT2/code/train.py` (lines around 1979-2026, `train_one_epoch` loop)

### Step 1: Write the diagnostic test

```python
# tests/test_p_pp_semantic_validation.py
class TestPPPSemanticValidation(unittest.TestCase):

    def test_alternating_loader_emits_both_sources(self):
        """Confirm alternating_loader yields P and PP batches with correct source tags."""
        from data import alternating_loader, SlidingWindowDataset, PP_TAG_P, PP_TAG_PP
        # Build minimal P and PP datasets from fixture data, check source tags alternate

    def test_pp_batch_produces_nonzero_recon_gradient(self):
        """A PP batch must produce a nonzero reconstruction gradient on model params."""
        # Forward a PP batch, compute recon loss, backward, check grad norm > 0

    def test_pp_batch_cls_loss_is_exactly_zero(self):
        """Classification loss contribution of a PP batch must be exactly 0."""
        from train import batch_loss_flags
        flags = batch_loss_flags(stage='B', batch_kind='PP')
        self.assertFalse(flags.get('cls', True),
                         "PP batch must not contribute to classification loss")

    def test_p_batch_cls_and_recon_both_nonzero(self):
        """A P batch in stage B must contribute to both cls and recon losses."""
        from train import batch_loss_flags
        flags = batch_loss_flags(stage='B', batch_kind='P')
        self.assertTrue(flags.get('cls', False))
        self.assertTrue(flags.get('recon', False))
```

Run: `python -m unittest tests.test_p_pp_semantic_validation`

### Step 2: Diagnose the zero-metrics issue

Check whether the issue is:
1. `batch_kind` is hardcoded to `"P"` even in alternating path (current: `batch_kind="P"` at call site line 2022 — but `train_one_epoch` overrides this via line 1086 when loader returns `(batch, source)` tuples)
2. PP loader is empty (no PP files loaded)
3. PP reconstruction loss weight is 0 or missing

Add per-kind metric accumulation inside `train_one_epoch`:
```python
p_recon_loss_accum = 0.0
pp_recon_loss_accum = 0.0
# ... in batch loop:
if active_batch_kind == 'PP':
    pp_recon_loss_accum += recon_loss.item()
else:
    p_recon_loss_accum += recon_loss.item()
# ... in return:
# add p_recon_mean and pp_recon_mean to stats
```

Log separately to TensorBoard: `train/p_recon_loss`, `train/pp_recon_loss`.

### Step 3: Fix any issues found and verify

Run a 5-epoch P+PP smoke:
```bash
python code/train.py --config code/config.p_pp_initial.yaml
```
Confirm in TensorBoard:
- `train/pp_recon_loss` > 0 every epoch
- `train/p_recon_loss` > 0 every epoch
- `train/cls_loss` > 0 every epoch (driven by P batches only)

---

## Task 3: Run formal P-only vs P+PP baseline experiment

**Goal:** First real experimental result. Does adding PP improve fatigue AUC?

**Files:**
- Use: `E:/GPTGAT2/code/config.p_only.yaml`
- Use: `E:/GPTGAT2/code/config.p_pp_full.yaml`
- Create: `E:/GPTGAT2/docs/results/2026-04-21-baseline-comparison.md`

### Experiment protocol

Run both configs for the same number of epochs (at minimum 30, ideally 50+):

```bash
# Run 1: P-only baseline
python code/train.py --config code/config.p_only.yaml \
  --run_name baseline_p_only

# Run 2: P+PP with domain adaptation off (domain_weight=0)
python code/train.py --config code/config.p_pp_full.yaml \
  --run_name baseline_p_pp_nodomain

# Run 3: P+PP with domain adaptation on (domain_weight=0.1)
python code/train.py --config code/config.p_pp_full.yaml \
  --run_name baseline_p_pp_domain01
```

### Metrics to compare

| Run | Val AUC | Val Acc | Train Loss | PP Recon Loss |
|-----|---------|---------|------------|---------------|
| P-only | — | — | — | N/A |
| P+PP no domain | — | — | — | — |
| P+PP domain=0.1 | — | — | — | — |

Record in `docs/results/2026-04-21-baseline-comparison.md`.

### Pass criterion

The experiment is successful if:
- P+PP val AUC >= P-only val AUC (or within noise margin with domain adaptation helping in later epochs)
- No collapse (collapse score stays < 0.9 throughout)
- PP recon loss is nonzero and decreasing

---

## Task 4: First ablation matrix

**Goal:** Produce the first ablation record for the current multi-dataset training setup.

**Variables to ablate (one axis at a time):**

### 4a. PP ratio (p_steps_per_pp)

| p_steps_per_pp | 1 | 2 | 4 | 8 |
|----------------|---|---|---|---|
| Val AUC | — | — | — | — |

Config change: `p_steps_per_pp: [1, 2, 4, 8]`

### 4b. Domain adaptation weight

| domain_weight | 0 | 0.01 | 0.1 | 0.5 | 1.0 |
|---------------|---|------|-----|-----|-----|
| Val AUC | — | — | — | — | — |

### 4c. Anti-forgetting weight

| anti_forgetting_weight | 0 | 0.01 | 0.1 |
|------------------------|---|------|-----|
| Val AUC | — | — | — |

### 4d. Decoder complexity (reconstruction head size)

Compare current decoder size vs. a shallower/deeper variant. Config: `decoder_hidden_dim`.

### Run script

```bash
for ratio in 1 2 4 8; do
  python code/train.py --config code/config.p_pp_full.yaml \
    --run_name ablation_ppratio_${ratio} \
    --override p_steps_per_pp=${ratio}
done
```

Note: if `--override` is not yet supported, add a minimal CLI override mechanism to `train.py`:
```python
# In argparse setup:
parser.add_argument('--override', nargs='*', default=[],
                    help='key=value config overrides')
# In config loading:
for kv in args.override:
    k, v = kv.split('=', 1)
    config[k] = yaml.safe_load(v)
```

### Deliverable

Record the ablation results in `docs/results/2026-04-21-baseline-comparison.md` or a follow-up results file under `docs/results/`.

---

## Summary of deliverables

| Task | Deliverable | Done when |
|------|-------------|-----------|
| Task 1 | `domain_weight` in train loop; `domain_loss` logged | test passes + nonzero in TensorBoard |
| Task 2 | `p_recon_loss` and `pp_recon_loss` logged separately | both nonzero in smoke run |
| Task 3 | `docs/results/2026-04-21-baseline-comparison.md` filled | 3 runs completed |
| Task 4 | Ablation results recorded | 4 ablation axes swept |

---

## Open questions (resolve before Task 3)

1. **latent_dim**: Does `model.latent_dim` exist, or does it need to be read from `config['latent_dim']`? Check `model.py` line ~194 for the bottleneck output shape.
2. **domain_labels from batch**: The batch `source` slot (index 3 in `collate_fn_multimodal`) holds string dataset names. Need `get_domain_id_mapping` to be called once before training with all known source names, then reused per-batch.
3. **latent shape for DomainAdaptationModule**: Model `latent` may be `(T, C)` or `(D,)`. Pool to `(D,)` before passing to `DomainDiscriminator`, which expects `(B, D)`.
