# Reproduction contract: native WebShop GRPO

## Identity and algorithm

Public SDAR: https://github.com/ZJU-REAL/SDAR at
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Recipe anchor: `examples/grpo_trainer/run_webshop_8gpu.sh`.
Entry: `verl.trainer.main_ppo.run_ppo`.
Loop: unchanged `verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit`.

This is pure GRPO, not SDAR/OPSD with a coefficient set to zero. The original
fixed reference model remains only for KL regularization. No Teacher,
privileged skill, retrieval, hint, SFT, API, Analyzer or RETHINK path is enabled.

Native advantages normalize outcome returns within task groups; clipped
policy loss, token mask, token-mean aggregation, entropy and invalid-action
penalty are unchanged. With sample token probability ratio r and advantage A,
the native objective uses the usual clipped surrogate,
`min(r*A, clip(r,1-epsilon,1+epsilon)*A)`, plus its original regularizers.
No reasoning-specific loss/mask is introduced.

## Exact shared recipe

- Two original Qwen2.5 Instruct sizes have fixed public revisions in models.json.
- 150 updates, each globally16 groups×8 episodes=128; dummy train parquet16
  rows supplies one dataloader batch per epoch, so150 epochs=150 updates.
  Dummy val parquet128 declares modality/count only. Tasks come from live env.
- LR1e-6, global decision-row mini64, per-GPU micro8, one PPO epoch,
  low_var_kl reference coefficient.01, entropy.001, invalid-action penalty.1.
- Prompt4096, response512, history2, max15 environment steps.
- Train temperature1/top_p1/top_k=-1; val temperature.4/do_sample=true/n1.
- env.rollout.n=8; model rollout.n=1. Native decision-row grouping and
  batch adjustment retained. Four FSDP ranks, two TP2 replicas, vLLM memory.6,
  chunkedprefill=false, eager=false, freecache=false.
- Initial validation and every5 updates, original128-episode sampler.
- Save20/final150, newest2 verified complete checkpoints; final HF export.

The original launcher requests8 GPUs; our four-GPU hardware binding changes
local mini64/8=8 to64/4=16 rows, while micro8 and global mini64 stay unchanged.
Native accumulation handles the difference. This does not promise bitwise
eight/four GPU equivalence. The1.5B model is a backbone extension;3B matches
the official script's model family and size.

## Minimal explicit adaptations

Only one native tracked source line changes: main_ppo imports our small
WebShopTrainer alias. That subclass calls super().fit and super()._validate.
It adds read-only local JSON/token/protocol telemetry and wraps only native
checkpoint save/load storage with atomic publication and post-save pruning.
No training loss, sampling, optimizer, rollout, environment manager, parser,
reward, prompt, memory, goal generator or Lucene search is patched. Public
source hashes and the tiny complete patch are enforced before launch.

Other portable changes: public pinned model/assets download, index build,
Python3.10/Java11 and compatible package constraints, project-relative paths,
including spaCy3.7.2's compatible preshed/cymem/murmurhash/srsly binary stack,
four-GPU Ray resource binding, console logger, explicit150-update stop and
local completion/export receipts. Do not inherit private proxies, conda paths
or destructive whole-machine cleanup commands from the original shell script.

## Environment/assets and limits of native evaluation

Preserve native `env.webshop.use_small=true`, `human_goals=false`,
`num_products=None`:1000 raw products, synthetic instructions, text observations.
Three JSON files and the small English spaCy model are SHA-pinned. The original
Google Drive endpoint was unavailable anonymously; both selected public HF
mirrors gave identical bytes for all three files. This is not a recovered
author-provided historical hash. Raw1000 may be filtered by native validity/
duplicate-ASIN checks; do not invent a different product pool.

Native conversion/indexing is used unchanged. num_products=None selects the
`search_engine/indexes` index, not indexes_1k. Build the original four index
directories. Receipt pins documents/corpus/dependency versions and local index
files. Lucene binary hashes can vary across independent index builds; compare
documents SHA and versions across models, not only index binary identity.
The accepted on-disk index must match its own receipt when resuming.

Each Ray worker owns its environment, RNG, shuffled goals, prices and searcher.
Training has128 actors and validation128 actors; do not pool them into a single
process because Python/NumPy/torch RNGs and goal shuffling are process-local.
Worker seed=base_seed+worker_index//group_n; val base_seed=train_seed+1000.
Native goal_idxs=train range(500,len(goals)), val range(500).
reset samples without replacement using each vector environment's persistent
RandomState and repeats by group_n.

Goals are shuffled independently inside workers. Numeric train/val index
separation is not proof of globally disjoint semantic goals, nor the standard
human500 split. Retain this native limitation and describe the dataset truthfully.
The telemetry previews the selection using a copied RNG state; it never consumes
or changes the native RNG. Continuous matched runs use identical index schedules.
Environment RNG/session state is not part of native checkpoints, so resume may
restart that schedule and is not an exact uninterrupted replay.

WebshopWorker gives sparse reward10 iff terminal raw score==1.0, otherwise0.
The continuous raw task_score is separately reported. Native manager supplies
`val/success_rate` and `val/webshop_task_score (not success_rate)`.
Do not compare score0-100 to success0-1 or call partial scores success.
Strict think/action format is read-only telemetry; original projection/allowed
actions/executor/feedback remain unchanged. Token IDs and unfiltered special
token text are retained.

The CPU environment smoke is a deterministic known-answer integration test
(search→product→options→buy→native reward) using goal metadata. It is not
policy evaluation, does not supply goals to Actor, and has no GPU updates.

## Storage and reproducibility acceptance

Native serializer/loader write four model, four optimizer, four extra_state
shards plus data.pt, config/tokenizer. A same-filesystem staging save disables
only native pre-save/in-memory deletion, validates torch.save ZIP/CRC and each
file SHA, writes CHECKPOINT_COMPLETE.json bound to recipe/source/model/assets/
release code/world_size, atomically publishes, verifies again, updates latest,
then prunes older verified same-run checkpoints. On restart scan disk, not
the lost in-memory path list; preserve invalid/foreign/linked/partial evidence.
No complete checkpoint means an error, never silent restart from zero.
Keep two complete saves plus a third in-progress peak and independent HF export.
Eight-rank checkpoints cannot be four-rank continuation.

GitHub clean CPU CI checks source/config/native gradients/storage and real
environment/index/reward. It does not establish CUDA memory fit, NCCL,
vLLM generation, real FSDP save/load or150-update scores. First hardware run
must confirm all of those, finite loss/grad, nonzero actual parameter updates,
intended model and4GPU topology. Report initial and final150 metrics; do not
pick a test-set best checkpoint. No benchmark numbers are prefilled.
