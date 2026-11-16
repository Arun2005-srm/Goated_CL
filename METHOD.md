# GOATED: Gated Old-teacher Anchored Teacher-Echo Distillation

Continual distillation for domain-incremental semantic segmentation
(Cityscapes → BDD100K → IDD by default). It merges the strengths of two papers
and adds three new parts:

| Source | What we keep | Its limitation |
|---|---|---|
| **CDTDD / SE2D** (Michel et al., 2026) | Sequence of teachers. The new student learns from the current teacher **and** the previous student (old-student KD on *external* data only). | (a) Every past teacher reaches the student only through a chain of students `T1→S1→S2→…`. Each hop is lossy, so in long sequences forgetting compounds (the paper's *Unseen Knowledge Forgetting*). (b) SE2D must know which samples are external to the teacher ("data-origin knowledge"), and the paper names this as a limitation. |
| **DoSe** (Reddy et al., WACV) | Batch-norm (BN) statistics as domain prototypes, BN-prototype self-distillation, a difficulty-ranked (VoG) exemplar buffer, stochastic restore. | (a) Classic teacher→student chain: once the new student becomes the teacher, the old teacher is gone. (b) Prototype evolution is a fixed-momentum running average, `P ← mP + (1-m)b`, compared with a plain Frobenius/MSE distance. |

## The idea in one paragraph

Old teachers cannot be kept: they are too large, or they are only available
through an API for a while. But we do not need the whole teacher. We need what
it **said** about a few representative images and what its **data looked
like**. So while teacher `T_t` is available, GOATED records:

1. **Teacher echoes.** For a small, difficulty-ranked exemplar buffer, we store
   compressed soft predictions of `T_t` (top-k probabilities at reduced
   resolution, a few KB per image).
2. **Statistics-aware BN domain prototypes.** A Kalman-filtered Gaussian belief
   over the per-image BN statistics of domain `t`, with mean, uncertainty and
   observation noise.
3. **Gaussian class prototypes.** Mean and variance of decoder features per
   (domain, class).

At every later stage, **all past teachers keep teaching at once**, through
their echoes. The current teacher and the old student join them. The three
signals are fused per pixel by confidence (entropy gating). The router that
decides where the old student matters is **data-free**: it compares each
image's BN statistics with the current teacher's own BN running statistics.
Those running statistics already describe the teacher's training data, so no
data-origin labels are needed.

## Components (each one is a boolean in `components:`)

| # | Switch | What it does | Origin |
|---|---|---|---|
| 1 | `teacher_kd` | Temperature KD from the current teacher `T_t` on the distillation batch | CDTDD |
| 2 | `old_student_kd` | KD from the frozen previous student `S_{t-1}`. It applies to *external* images only (SE2D, `old_student_on: external`) or to all images (self-distillation, `all`) | CDTDD |
| 3 | `exemplar_buffer` | Per-stage priority buffer (VoG / entropy / random), mixed into each step | DoSe |
| 4 | `teacher_echo` | Stores `T_t`'s top-k soft labels with each exemplar, so past teachers keep supervising | **new** |
| 5 | `bn_layer_distill` | Per-image BN statistics of the student match the old student's (DoSe Eq. 6) | DoSe |
| 6 | `bn_domain_prototypes` | Student BN statistics are pulled towards the prototype of the image's domain, including replayed old domains (DoSe Eq. 7) | DoSe |
| 7 | `stats_aware_prototypes` | Replaces the running average with a Kalman/Welford estimator and a Mahalanobis distance; off falls back to DoSe's EMA + MSE | **new** |
| 8 | `auto_domain_routing` | Soft "is this image external to `T_t`?" score from the teacher's own BN running statistics. Replaces SE2D's data-origin labels | **new** |
| 9 | `entropy_gated_fusion` | Per-pixel confidence-weighted fusion of {current teacher, old student, echoes} into one target; off sums the KD terms with fixed weights | **new** |
| 10 | `class_prototype_replay` | Samples features from old (domain, class) Gaussians and trains the classifier head on them, with no images needed | prototype replay |
| 11 | `stochastic_restore` | Bernoulli restore of weights to the stage-start student | DoSe / CoTTA |
| 12 | `aug_pseudo_label` | CE on flip-averaged teacher pseudo-labels (DoSe Eq. 8–9) | DoSe |
| 13 | `supervised_ce` | Ground-truth CE (turns CD into supervised DIL; off by default) | baseline |

Turning off components gives the baselines as presets in `scripts/run_ablations.py`:
KL-only, self-distillation, SE2D, DoSe-style, and the full model. A
leave-one-out sweep is one command (`scripts/run_ablations.py`).

## Statistics-aware prototype evolution (component 7)

For every tracked BN layer and channel we observe per-image statistics
`b = (μ, log σ²)` of the BN input. DoSe uses `P ← 0.9 P + 0.1 b`. That rule
has four problems:

* **Same trust for every channel.** A channel whose statistics jump around from
  image to image moves the prototype as much as a stable channel.
* **No notion of confidence.** Early in a domain the prototype is built from
  three images, yet it is trusted as much as one built from three thousand.
* **No outlier protection.** One odd batch (a tunnel, a dark frame, a
  mis-routed external image) moves the prototype by 10%.
* **Distance ignores scale.** The MSE is dominated by high-variance channels and
  deep layers, so most layers barely contribute.

GOATED treats each prototype channel as a 1-D Kalman filter:

```
R  = Var_welford(observations)               # observation noise, measured from data
P⁻ = P + q·R                                 # small process noise lets the prototype drift slowly
NIS = mean_c (b − m)² / (P⁻ + R)             # normalized innovation (Mahalanobis)
ω  = min(1, gate / NIS)                      # robust down-weighting of outlier batches
K  = P⁻ / (P⁻ + R/ω)                         # stats-aware gain, per channel
m ← m + K (b − m);   P ← (1 − K) P⁻
```

For the first `warmup` observations it uses the exact cumulative mean (`K =
1/n`), which is the minimum-variance estimate when nothing is known yet. After
that the gain adapts on its own:

* Noisy channels have large `R`, so a small `K`.
* A confident prototype has small `P`, so a small `K`.
* A true slow drift accumulates `q·R` and stays trackable.
* Outliers are gated.

The prototype **loss** works in standard-deviation units,
`z = (s − m) / sqrt(R̃ + P)`, which puts every channel and every layer on the
same scale. It has two safeguards, and both came out of the smoke test (see
below):

* **Variance shrinkage.** `R̃ = (1−a)R + a·mean(R)` per layer, kept separate for
  the mean and log-variance halves. Without it, near-constant channels give
  `1/sqrt(R) → ∞` gradients.
* **Hinge at the natural spread.** The loss is `huber(relu(|z| − κ))` with
  `κ = 2`. It penalizes only statistics that leave the domain's normal
  image-to-image range, instead of pulling every image to the mean.
  EMA + MSE (DoSe) cannot do this because it does not know the spread.

`bn_layer_distill` (student vs. old student) is applied to **replayed
old-domain images only**. DoSe's θ_{k−1} is the previous *iteration*, so it only
smooths training. Anchoring current-domain images to the previous *stage's*
student blocks adaptation to the new domain, and in the smoke test that
collapsed the final stage.

## Why this should reduce forgetting

1. **Supervision is no longer only a chain.** In CDTDD the knowledge of `T_1`
   reaches stage `t` only through `t−1` imperfect copies, so its error grows
   with sequence length. Echoes give a **direct** path from every `T_j` to every
   later student. Echo error is bounded by the compression (top-k and spatial
   pooling) and does **not grow with `t`**. This is the same argument that makes
   DER-style logit replay beat plain replay, applied to teachers instead of the
   student's own past logits. Under this argument, forgetting in long sequences
   is no longer unavoidable.
2. **Each source teaches where it is competent.** The current teacher is
   confident on its own domain and uncertain on images outside it. CDTDD's
   appendix shows this: teacher entropy is a good proxy for whether it knows
   the data. Entropy gating hands those pixels to the old student or to the
   echo. SE2D does this with a hard, label-dependent mask. GOATED does it
   softly, per pixel, and without labels.
3. **Data-free routing.** A BN running mean and variance summarize the data the
   teacher was trained on, which makes them free prototypes of the teacher's
   domain. The router computes the KL divergence between an image's BN
   statistics and the teacher's running statistics, calibrated on the teacher's
   own domain. This answers SE2D's "requires data-origin knowledge" limitation.
4. **Representation anchoring at two levels.** BN-prototype losses keep
   low-level feature statistics of old domains in place (DoSe's insight). Class
   Gaussian replay keeps the classifier boundaries for old-domain feature
   distributions. Both are cheap and need no stored teacher.
5. **Every addition is optional and isolated.** If a part does not help on your
   data, the ablation shows it directly.

## Honest caveats

* This is a research hypothesis with a working implementation. It is **not** a
  validated result. The numbers come from running the ablations.
* Echoes hold soft labels for buffer images only. Coverage of old domains is
  bounded by the buffer size: 20 images per stage by default, configurable.
* Class Gaussian prototypes live in the student's feature space, which drifts.
  The BN anchoring reduces that drift but does not remove it, so the default
  weight is small.
* The router assumes the teacher has BatchNorm layers. All bundled encoders
  have them. ViT/LayerNorm teachers would need a different prototype source.
* **The router is weak for multi-domain teachers.** If a teacher was trained on
  two domains, its BN running stats describe a *mixture*. In the smoke test the
  calibration spread was wide, and the external domain scored 0.03 while an
  internal domain scored 0.09. With single-domain teachers (`driving_dil`), it
  separates as intended. Check `routing_mean_external_score` in
  `diagnostics.json` on your data. A per-domain Gaussian-mixture calibration
  is the natural next step.

## Smoke-test evidence (synthetic, NOT a result)

Setup: three synthetic domains (sunny → foggy → night), tiny random-init
CNNs, 8 epochs per stage, one seed. This checks that the parts work and
interact sanely. It says nothing about Cityscapes/BDD/IDD.

| run | sunny | foggy | night | avg mIoU | forgetting |
|---|---|---|---|---|---|
| KL only (CDTDD baseline) | 29.1 | 14.6 | 60.8 | 34.8 | 40.8 |
| SE2D | 28.6 | 14.6 | 60.9 | 34.7 | 41.0 |
| GOATED full | 59.7 | 46.2 | 45.4 | 50.4 | 11.9 |
| − exemplar buffer (no echo / replay) | 26.0 | 33.8 | 59.0 | 39.6 | 33.0 |
| − old-student KD | 63.2 | 53.8 | 55.3 | 57.4 | 4.2 |
| − entropy fusion | 63.2 | 48.3 | 48.4 | 53.3 | 7.3 |

What this shows:

1. Replay plus teacher echoes is the main source of retention: 3–4× less
   forgetting than KL or SE2D.
2. The full model gives up plasticity on the last domain (night 45 vs 61). On
   this toy data, several single components cost more than they return. The
   real-data ablation table decides what stays on. That is why every part is a
   switch.
3. SE2D equals KL here, as it should. In pure DIL the distillation data
   contains no external images, so the SE2D mask is always empty. Use
   `driving_cd.yaml` to test SE2D-style external data.
