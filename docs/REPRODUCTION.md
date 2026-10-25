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
  Dummy val parquet140 declares modality/count only. Tasks come from live env.
- LR1e-6, global decision-row mini64, per-GPU micro8, one PPO epoch,
  low_var_kl reference coefficient.01, entropy.001, invalid-action penalty.1.
- Prompt4096, response512, history2, max15 environment steps.
- Train temperature1/top_p1/top_k=-1; val temperature.4/do_sample=true/n1.
- env.rollout.n=8; model rollout.n=1. Native decision-row grouping and
  batch adjustment retained. Four FSDP ranks, two TP2 replicas, vLLM memory.6,
  chunkedprefill=false, eager=false, freecache=false.
- Initial validation and every5 updates, all of the same fixed140 disjoint tasks.
- Save20/final150, newest2 verified complete checkpoints; final HF export.

The original launcher requests8 GPUs; our four-GPU hardware binding changes
local mini64/8=8 to64/4=16 rows, while micro8 and global mini64 stay unchanged.
Native accumulation handles the difference. This does not promise bitwise
eight/four GPU equivalence. The1.5B model is a backbone extension;3B matches
the official script's model family and size.

## Minimal explicit adaptations

Two native source files are patched. main_ppo imports our WebShopTrainer
alias, which calls super().fit and super()._validate for JSON/token telemetry,
fixed validation reset and transactional checkpoint storage. The WebShop
vector factory instantiates a small subclass that filters each worker's
actual goal catalog by shared content IDs before Actor interaction; its index
range then spans that already filtered catalog. This is necessary because
different workers independently shuffle their catalogs.
No training loss, optimizer, rollout, environment manager, parser, reward,
prompt, memory, goal generator or Lucene search is changed. Native training
group sampling remains uniform without replacement, on the new disjoint
training pool. Both changed files and all locked unchanged files are verified.

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
Training has128 actors and validation140 actors; do not pool them into a single
process because Python/NumPy/torch RNGs and goal shuffling are process-local.
Worker seed=base_seed+worker_index//group_n; val base_seed=train_seed+1000.
The original implementation uses train indices500..N-1 and validation0..499;
indices from different shuffled catalogs do not guarantee disjoint actual
tasks. This release instead filters actual goals first, then uses local
indices0..len(filtered_pool)-1. Training samples without replacement with
the original persistent RandomState and repeats by group_n.

### Fixed140 content-based split

configs/task_split.json fixes140 unique visible objectives:28 each from the
native beauty/electronics/fashion/garden/grocery categories, with140 different
target ASINs,123 query types and126 product-category paths. Selection balances
attribute/option counts, then query/path diversity, using a fixed2026 SHA tie
break. It does not use model output, rewards or success rates. Rebuild it with
select_validation/make_manifest in src/task_split.py on the unmodified native
seed1000 catalog; env_smoke.py verifies equivalent manifest content.

A content ID hashes normalized instruction, attributes and options, ignoring
only the generated price suffix and target ASIN aliases. Native workers generate
different price thresholds, so these variants must not bypass the exclusion.
Of6910 native entries,145 map to the140 held-out IDs;6765 remain for training.
Training entries are otherwise kept, including their original duplicates,
order and goal weights. This is objective-held-out, not product-held-out:
other requirements for the same products may still be used in training.

Every worker filters its actual server.goals before any Actor observation,
rebuilds native goal weights, discards the unfiltered constructor session,
and checks allowed IDs on reset/step, including native terminal auto-resets.
The raw catalog count and content-set hash must match the public lock.

configs/fixed_validation140.json binds all140 held-out IDs to worker seeds
1000..1139 and goal indices0..139. Every evaluation runs one native140-episode
batch, exactly once per task. A temporary validation-only deterministic chooser
uses that order, then restores the original private RNG object even on failure.
Actual reset task IDs must match before generation. The global training/model
RNG and original training sampler are not replaced.

For all validation updates the task set is constant:

\[
\mathcal{V}_t=\mathcal{V}_0,\qquad t\in\{0,5,10,\ldots,150\}.
\]

After the first actual reset, persist manifest/selection/initial-observation SHA
in fixed_validation_identity.json. Every subsequent validation, including resume,
must match the same initial observations, or fail before policy generation.
The summary carries this identity so two sizes can verify task equality. Source,
assets and package locks also constrain per-worker goals and prices. Each task
has a distinct content ID, and the pool invariant is

\[
|\mathcal V|=140,\qquad \mathcal T\cap\mathcal V=\varnothing.
\]

This is a custom fixed synthetic split, not WebShop's canonical human500,
unseen-product or strict cross-domain zero-shot benchmark.

Validation remains one complete policy rollout/task, temperature.4 with sampling,
15 steps/history2/response512, reporting native success and continuous task score.
Fixed tasks do not make generation deterministic; no claim of identical scores
across repeated evaluations. Evaluation tasks are never fed to an optimizer.

Native checkpoints do not snapshot WebShop environment/session RNG state. Fixed
validation identities survive resume, but training remains stochastic and is not
an exact uninterrupted replay. Older random/128-validation run bindings are deliberately
incompatible; preserve their code/output and use a new run directory for this recipe.

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
