# Cross-Domain Training Design

**Goal:** Add cross-domain learning to the current multi-dataset training project without coupling the domain objective into the TGATUNet model definition.

**Decision:** Use the existing `DomainAdaptationModule` as an external training head on latent representations. `TGATUNet` keeps producing reconstruction, logits, and latent features; `train_one_epoch` computes the domain objective from `latents` and stable dataset `source_ids`.

## Scope

- Current project only: multi-dataset P+PP training.
- Old manuscript or Overleaf work is out of scope.
- Domain count follows config: 5 P datasets + 3 PP datasets = 8 domains.

## Architecture

1. Data loading assigns a stable global source id per dataset name.
2. The model forward path returns latent features for each window.
3. Training computes reconstruction/classification losses as before.
4. When enabled, `DomainAdaptationModule` receives `(latents, source_ids)` and returns adversarial/MMD/CORAL losses.
5. Total loss adds `domain_weight * domain_losses["total"]`.

## Batch Routing

- P batches can use reconstruction, classification improvement, and domain adaptation.
- PP batches use reconstruction and domain adaptation, but classification routing is disabled by default.
- Mixed batches are allowed; classification remains controlled by helper flags and existing aux-classifier availability.

## Minimal Implementation

- Add stage helpers in `code/train.py`: `select_stage`, `get_training_stage`, `stage_loss_weights`, `batch_loss_flags`, and `route_classification_for_batch`.
- Extend `train_one_epoch` with `domain_module`, `domain_weight`, and `n_p_sources` parameters.
- Track total domain loss plus P and PP reconstruction metrics in the training return tuple.
- Prefer `model.forward_batch()` when available, falling back to per-window model calls for existing models.
- Fix source inference in `code/data.py` so each configured dataset maps to a stable domain id.

## Verification

Target tests:

- `python -m unittest code.tests.test_p_pp_domain_and_forgetting -v`
- `python -m unittest code.tests.test_p_pp_stage_schedule code.tests.test_p_pp_batch_loss_flags code.tests.test_p_pp_loss_routing -v`
- Existing multi-dataset config tests after integration.
