# GPTGAT2 Codebase Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize project paths and separate source, plans, runbooks, tests, configs, and generated artifacts without breaking the current flat-module entrypoints.

**Architecture:** Keep `code/` runnable in its current flat form, then introduce a small shared path layer and compatibility-friendly file moves around it. Reorganize docs and low-risk folders first; leave deep module splitting for a later pass once imports and runtime paths are stable.

**Tech Stack:** Python, PyTorch, YAML configs, unittest, Markdown docs.

---

### Task 1: Reclassify Existing Planning Documents

**Files:**
- Create: `E:/GPTGAT2/docs/plans/active/.gitkeep`
- Create: `E:/GPTGAT2/docs/plans/archive/.gitkeep`
- Create: `E:/GPTGAT2/docs/runbooks/.gitkeep`
- Modify: `E:/GPTGAT2/docs/plans/2026-04-21-phase2-domain-experiments.md`
- Move: `E:/GPTGAT2/docs/plans/2026-03-15-model-p-pp-training-implementation.md`
- Move: `E:/GPTGAT2/docs/plans/2026-03-15-model-p-pp-training-runbook.md`
- Create: `E:/GPTGAT2/docs/README.md`

**Step 1: Write the failing test**

```python
def test_docs_layout_separates_active_archive_and_runbooks():
    assert Path("docs/plans/active").is_dir()
    assert Path("docs/plans/archive").is_dir()
    assert Path("docs/runbooks").is_dir()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest code.tests.test_docs_layout -v`
Expected: FAIL because the new doc layout and index file do not exist yet.

**Step 3: Write minimal implementation**

- Create the new doc directories.
- Move the completed 2026-03-15 implementation plan into `archive`.
- Move the runbook into `docs/runbooks`.
- Leave the 2026-04-21 phase-2 plan as the active experiment plan.
- Add a short `docs/README.md` that explains where active plans, archived plans, and runbooks live.

**Step 4: Run test to verify it passes**

Run: `python -m unittest code.tests.test_docs_layout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs
git commit -m "docs: separate active plans, archive, and runbooks"
```

### Task 2: Add Project-Root Path Resolution

**Files:**
- Create: `E:/GPTGAT2/code/project_paths.py`
- Create: `E:/GPTGAT2/code/tests/test_project_paths.py`
- Modify: `E:/GPTGAT2/code/data.py`
- Modify: `E:/GPTGAT2/code/train.py`
- Modify: `E:/GPTGAT2/code/downstream_eval.py`
- Modify: `E:/GPTGAT2/code/paper_downstream_pipeline.py`
- Modify: `E:/GPTGAT2/code/ablation_experiments.py`
- Modify: `E:/GPTGAT2/code/sensitivity_analysis.py`
- Modify: `E:/GPTGAT2/code/export_completed.py`

**Step 1: Write the failing test**

```python
def test_relative_artifact_paths_resolve_from_project_root():
    cfg = {"log_dir": "Logs", "checkpoint_dir": "Checkpoints", "tensorboard_dir": "runs"}
    normalized = normalize_config_paths(cfg)
    assert str(normalized["log_dir"]).endswith("artifacts/logs")
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest code.tests.test_project_paths -v`
Expected: FAIL because no shared path-normalization helper exists yet.

**Step 3: Write minimal implementation**

- Add a shared helper that resolves project root from `code/`.
- Normalize legacy relative artifact names like `Logs`, `Checkpoints`, `runs`, and `Data/cache` into stable root-level `artifacts/...` locations.
- Normalize `data_dir: Data` to the repo-root data directory regardless of the current working directory.
- Keep compatibility with existing config keys and script defaults.

**Step 4: Run test to verify it passes**

Run: `python -m unittest code.tests.test_project_paths -v`
Expected: PASS

**Step 5: Commit**

```bash
git add code/project_paths.py code/tests/test_project_paths.py code/data.py code/train.py code/downstream_eval.py code/paper_downstream_pipeline.py code/ablation_experiments.py code/sensitivity_analysis.py code/export_completed.py
git commit -m "refactor: stabilize project-root path resolution"
```

### Task 3: Introduce Config and Test Subfolders Without Breaking Imports

**Files:**
- Create: `E:/GPTGAT2/code/configs/base.yaml`
- Create: `E:/GPTGAT2/code/configs/modes/p_only.yaml`
- Create: `E:/GPTGAT2/code/configs/modes/p_pp_initial.yaml`
- Create: `E:/GPTGAT2/code/configs/modes/p_pp_full.yaml`
- Create: `E:/GPTGAT2/code/configs/presets/smoke.yaml`
- Create: `E:/GPTGAT2/code/configs/presets/smoke_numworkers4_bs256.yaml`
- Create: `E:/GPTGAT2/code/configs/presets/paper_quick.yaml`
- Create: `E:/GPTGAT2/code/configs/presets/paper_tiny.yaml`
- Create: `E:/GPTGAT2/code/tests/model/.gitkeep`
- Create: `E:/GPTGAT2/code/tests/training/.gitkeep`
- Create: `E:/GPTGAT2/code/tests/reprocess/.gitkeep`
- Create: `E:/GPTGAT2/code/tests/experiments/.gitkeep`
- Modify: `E:/GPTGAT2/code/config.yaml`
- Modify: `E:/GPTGAT2/code/config.p_only.yaml`
- Modify: `E:/GPTGAT2/code/config.p_pp_initial.yaml`
- Modify: `E:/GPTGAT2/code/config.p_pp_full.yaml`
- Modify: `E:/GPTGAT2/code/config.smoke.yaml`
- Modify: `E:/GPTGAT2/code/config.smoke_numworkers4_bs256.yaml`
- Modify: `E:/GPTGAT2/code/config.paper_quick.yaml`
- Modify: `E:/GPTGAT2/code/config.paper_tiny.yaml`

**Step 1: Write the failing test**

```python
def test_config_layout_exposes_modes_and_presets():
    assert Path("code/configs/modes/p_only.yaml").exists()
    assert Path("code/configs/presets/smoke.yaml").exists()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest code.tests.test_docs_layout -v`
Expected: FAIL because the config subfolders do not exist yet.

**Step 3: Write minimal implementation**

- Copy the current canonical configs into `code/configs/...`.
- Keep existing top-level config files as compatibility entrypoints for old commands.
- Create empty grouped test directories so the suite has an obvious future layout target.

**Step 4: Run test to verify it passes**

Run: `python -m unittest code.tests.test_docs_layout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add code/configs code/tests code/config*.yaml
git commit -m "chore: add grouped config and test directories"
```

### Task 4: Add Prep and Experiment Folder Targets With Compatibility Wrappers

**Files:**
- Create: `E:/GPTGAT2/code/prep/raw_to_unified/.gitkeep`
- Create: `E:/GPTGAT2/code/prep/unified_to_p/.gitkeep`
- Create: `E:/GPTGAT2/code/prep/to_pp/.gitkeep`
- Create: `E:/GPTGAT2/code/experiments/.gitkeep`
- Modify: `E:/GPTGAT2/code/.gitignore`
- Create: `E:/GPTGAT2/code/tests/test_layout_targets.py`

**Step 1: Write the failing test**

```python
def test_code_layout_has_prep_and_experiments_targets():
    assert Path("code/prep/raw_to_unified").is_dir()
    assert Path("code/prep/unified_to_p").is_dir()
    assert Path("code/prep/to_pp").is_dir()
    assert Path("code/experiments").is_dir()
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest code.tests.test_layout_targets -v`
Expected: FAIL because the target layout directories do not exist yet.

**Step 3: Write minimal implementation**

- Add the target directories for future script grouping.
- Update `.gitignore` so root-level `artifacts/` and tool caches are ignored explicitly.
- Do not move core Python modules yet; just prepare the layout safely.

**Step 4: Run test to verify it passes**

Run: `python -m unittest code.tests.test_layout_targets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add code/prep code/experiments code/tests/test_layout_targets.py code/.gitignore
git commit -m "chore: prepare stable code layout targets"
```

### Task 5: Verify Reorganization Safety

**Files:**
- Test: `E:/GPTGAT2/code/tests/test_project_paths.py`
- Test: `E:/GPTGAT2/code/tests/test_docs_layout.py`
- Test: `E:/GPTGAT2/code/tests/test_layout_targets.py`
- Test: `E:/GPTGAT2/code/tests/test_p_pp_loader_setup.py`
- Test: `E:/GPTGAT2/code/tests/test_p_pp_data_contract.py`
- Test: `E:/GPTGAT2/code/tests/test_downstream_eval.py`

**Step 1: Write the failing test**

```python
def test_existing_import_style_still_works():
    import train
    import data
    import model
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest code.tests.test_layout_targets -v`
Expected: FAIL if the reorganization broke the current flat import contract.

**Step 3: Write minimal implementation**

- Keep the flat import contract intact.
- Verify that the current training/eval modules still import and key tests still pass.
- Do not deepen the refactor unless these guards remain green.

**Step 4: Run test to verify it passes**

Run: `python -m unittest code.tests.test_project_paths code.tests.test_docs_layout code.tests.test_layout_targets code.tests.test_p_pp_loader_setup code.tests.test_p_pp_data_contract code.tests.test_downstream_eval -v`
Expected: PASS

**Step 5: Commit**

```bash
git add code/tests
git commit -m "test: verify reorganization compatibility"
```
