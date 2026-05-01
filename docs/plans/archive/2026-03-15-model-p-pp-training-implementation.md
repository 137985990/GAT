# GPTGAT2 P/PP Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add safe P/PP mixed training to GPTGAT2 so trusted P datasets supervise fatigue classification while PP datasets improve reconstruction and domain robustness without label leakage.

**Architecture:** Keep the current reconstruction-centric TGATUNet backbone and aux-classifier workflow, but add explicit P/PP routing, alternating loaders, staged training, and task-aware completion losses. Treat PP as auxiliary-only: reconstruction and domain alignment yes, classification no.

**Tech Stack:** Python, PyTorch, existing GPTGAT2 codebase (`train.py`, `data.py`, `model.py`, `aux_classifier.py`, `domain_adaptation.py`), unittest.

---

### Task 1: Define P/PP data contract and alternating loaders

**Files:**
- Modify: `E:/GPTGAT2/code/data.py`
- Modify: `E:/GPTGAT2/code/config.yaml`
- Test: `E:/GPTGAT2/code/tests/test_p_pp_data_contract.py`

**Step 1: Write the failing test**

```python
def test_p_pp_batch_contract_and_source_uniqueness():
    dataset = create_mixed_dataset(...)
    sample = dataset[0]
    assert len(sample) >= 8
    assert hasattr(dataset, 'dataset_kind_map')
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_p_pp_data_contract`
Expected: FAIL because mixed P/PP contract and loader helpers do not exist yet.

**Step 3: Write minimal implementation**

- Add explicit dataset-kind metadata (`P` or `PP`) to dataset construction.
- Add validation that a source cannot appear in both P and PP pools in the same run.
- Add alternating loader helper with configurable `p_steps_per_pp_step`.
- Ensure labels from PP are absent/ignored rather than encoded as fake zeros.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_p_pp_data_contract`
Expected: PASS

**Step 5: Commit**

```bash
git add code/tests/test_p_pp_data_contract.py code/data.py code/config.yaml
git commit -m "feat: add P and PP data loading contract"
```

### Task 2: Add staged hybrid loss routing in training loop

**Files:**
- Modify: `E:/GPTGAT2/code/train.py`
- Modify: `E:/GPTGAT2/code/aux_classifier.py`
- Test: `E:/GPTGAT2/code/tests/test_p_pp_loss_routing.py`

**Step 1: Write the failing test**

```python
def test_pp_batch_skips_classification_loss():
    losses = compute_losses(batch_kind='PP', ...)
    assert losses['cls'] == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_p_pp_loss_routing`
Expected: FAIL because the current train loop has no P/PP-specific routing.

**Step 3: Write minimal implementation**

- Stage A: `L_recon + L_domain` on alternating P/PP batches.
- Stage B: `P -> L_recon + L_task_recon + L_cls_improve + L_domain`; `PP -> L_recon + L_domain`.
- Keep `aux_classifier` as the task anchor for completed-vs-original improvement.
- Add explicit assertions that PP never computes label-based losses.
- Add configurable warmup epochs and loss ramps.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_p_pp_loss_routing`
Expected: PASS

**Step 5: Commit**

```bash
git add code/tests/test_p_pp_loss_routing.py code/train.py code/aux_classifier.py
git commit -m "feat: add staged P PP hybrid loss routing"
```

### Task 3: Integrate domain adaptation and anti-forgetting safeguards

**Files:**
- Modify: `E:/GPTGAT2/code/train.py`
- Modify: `E:/GPTGAT2/code/domain_adaptation.py`
- Test: `E:/GPTGAT2/code/tests/test_p_pp_domain_and_forgetting.py`

**Step 1: Write the failing test**

```python
def test_domain_loss_runs_on_p_and_pp_only():
    stats = train_step(...)
    assert 'domain' in stats
    assert 'ewc' in stats or 'teacher' in stats
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_p_pp_domain_and_forgetting`
Expected: FAIL because domain adaptation is not wired into `train.py` and anti-forgetting does not exist.

**Step 3: Write minimal implementation**

- Wire `DomainAdaptationModule` into the training loop using latent features.
- Use dataset-level domain IDs, not just coarse P/PP IDs.
- Add conservative PP scheduling and supervised anchor floor.
- Add one anti-forgetting mechanism: EMA teacher or representation anchor on P.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_p_pp_domain_and_forgetting`
Expected: PASS

**Step 5: Commit**

```bash
git add code/tests/test_p_pp_domain_and_forgetting.py code/train.py code/domain_adaptation.py
git commit -m "feat: add domain alignment and anti-forgetting"
```

### Task 4: Add metrics for collapse, completion benefit, and ablations

**Files:**
- Modify: `E:/GPTGAT2/code/train.py`
- Create: `E:/GPTGAT2/code/tests/test_p_pp_metrics.py`
- Modify: `E:/GPTGAT2/PROJECT_STATUS.md`

**Step 1: Write the failing test**

```python
def test_training_stats_include_completion_and_collapse_metrics():
    stats = build_train_stats(...)
    assert 'cls_gap_real_vs_completed' in stats
    assert 'latent_effective_rank' in stats
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_p_pp_metrics`
Expected: FAIL because these metrics are not logged today.

**Step 3: Write minimal implementation**

- Add `real vs completed` classification gap metrics.
- Add latent variance/effective-rank/cosine concentration monitoring.
- Add per-dataset P metrics and PP diagnostics separately.
- Document required ablations in `PROJECT_STATUS.md`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_p_pp_metrics`
Expected: PASS

**Step 5: Commit**

```bash
git add code/tests/test_p_pp_metrics.py code/train.py PROJECT_STATUS.md
git commit -m "feat: add P PP monitoring and ablation metrics"
```

### Task 5: Add UV-based reproducible experiment environment and smoke run

**Files:**
- Create: `E:/GPTGAT2/uv.toml`
- Create: `E:/GPTGAT2/.python-version`
- Create: `E:/GPTGAT2/docs/plans/2026-03-15-model-p-pp-training-runbook.md`

**Step 1: Write the failing test**

```python
def test_runbook_mentions_uv_setup_and_4070ti_smoke_run():
    text = Path('docs/plans/2026-03-15-model-p-pp-training-runbook.md').read_text()
    assert 'uv' in text.lower()
    assert '4070ti' in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_environment_runbook`
Expected: FAIL because the runbook and uv config do not exist.

**Step 3: Write minimal implementation**

- Add UV environment bootstrap notes and pinned Python version.
- Add runbook for local smoke tests on the user's 4070 Ti.
- Include tiny sanity experiments before full training.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_environment_runbook`
Expected: PASS

**Step 5: Commit**

```bash
git add uv.toml .python-version docs/plans/2026-03-15-model-p-pp-training-runbook.md
git commit -m "docs: add uv environment and local experiment runbook"
```
