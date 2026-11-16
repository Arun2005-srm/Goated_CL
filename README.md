# Goated_CL

Continual distillation for domain-incremental semantic segmentation. It combines
**CDTDD/SE2D** (a sequence of teachers plus the old student) and **DoSe** (BN
prototypes plus a difficulty-ranked replay buffer), and adds teacher echoes,
statistics-aware prototypes, data-free external-data routing and entropy-gated
fusion. The method and the reasons it should work are in **[METHOD.md](METHOD.md)**.

Default protocol: **Cityscapes → BDD100K → IDD**, with one teacher per domain.
Each teacher is unavailable after its stage.

## Quick start

```bash
pip install -r requirements.txt          # install a CUDA build of torch/torchvision first
python scripts/make_toy_dataset.py       # synthetic data, offline
python train.py --config configs/smoke.yaml
python -m pytest -q tests
```

On real data, set the three dataset roots in [configs/driving_dil.yaml](configs/driving_dil.yaml)
(or pass them with `--set`), then run:

```bash
python train.py --config configs/driving_dil.yaml --only eda      # look at the data first
python train.py --config configs/driving_dil.yaml                 # EDA -> teachers -> distillation -> reports
python train.py --config configs/driving_dil.yaml --set domains.idd.root=D:/data/IDD_Segmentation
python scripts/run_ablations.py --config configs/driving_dil.yaml # baselines + leave-one-out table
```

`train.py` runs these steps:

1. Validate the config.
2. Discover image/mask pairs per domain.
3. Apply seeded, group-disjoint hold-outs and check for split leakage.
4. Run the optional resize cache.
5. Run EDA.
6. Train or load the teachers.
7. Run continual distillation.
8. Write the reports.

Use `--only validate|eda|teachers` to stop early and `--resume runs/.../last.pt` to
continue a run.

## Datasets

| Config | Layout expected under `root` | Notes |
|---|---|---|
| [cityscapes.yaml](configs/datasets/cityscapes.yaml) | `leftImg8bit/`, `gtFine/*_labelIds.png` | labelIds → 19 trainIds. Official val = our test. 10% of train cities = val |
| [bdd100k.yaml](configs/datasets/bdd100k.yaml) | `10k/{train,val}/*.jpg`, `labels/{train,val}/*_train_id.png` | Already 19 train IDs. Same layout as your `bekerly` folder |
| [idd.yaml](configs/datasets/idd.yaml) | `leftImg8bit/`, `gtFine/*_gtFine_labellevel3Ids.png` | Run IDD's `createLabels.py --id-type level3Id` first. Level-3 → Cityscapes-19 mapping is documented in the file |

To add any other dataset, write a YAML file with `root`, `label_map` (or
`rgb_label_map`), `splits` (glob templates or CSV manifests) and an optional
`holdout`. Then reference it from `domains:`. No Python is needed. The EDA
report lists any raw label value that the map sends to *ignore*, so you can
check a mapping on the first run.

**Hardware.** The defaults (ResNet-50 teacher, MobileNetV3 student, 384×768,
batch 4, AMP) fit a 4 GB GPU. For 2K images, set `data.cache.enabled: true`. It
resizes everything once to `data.cache.size`, which makes loading much faster.

## Modular components (ablations)

Every component is a boolean under `components:`. A component that is off
builds no buffers, runs no forward pass and adds no loss.
`model_components.json` records what was actually constructed.

| Switch | Origin | Switch | Origin |
|---|---|---|---|
| `teacher_kd` | CDTDD | `stats_aware_prototypes` | new |
| `old_student_kd` | CDTDD / SE2D | `auto_domain_routing` | new |
| `exemplar_buffer` | DoSe | `entropy_gated_fusion` | new |
| `teacher_echo` | new | `class_prototype_replay` | prototype replay |
| `bn_layer_distill` | DoSe | `stochastic_restore` | DoSe / CoTTA |
| `bn_domain_prototypes` | DoSe | `aug_pseudo_label` | DoSe |
| `supervised_ce` | baseline | | |

```bash
python train.py --config configs/driving_dil.yaml --set components.teacher_echo=false --set output_dir=runs/no_echo
```

`scripts/run_ablations.py` runs these presets: `kl_only`, `self_distillation`,
`se2d`, `dose_style`, `finetune_supervised`, `goated_full` and every `no_<component>`.
Teachers are trained once, in `teachers.dir`, and reused by all runs. The
script writes `ablation_table.md`.

## Protocols

* [driving_dil.yaml](configs/driving_dil.yaml): stage *t* uses teacher *t* and
  distils on domain *t*. This is the DoSe-style pair setting.
* [driving_cd.yaml](configs/driving_cd.yaml): CDTDD-style. Cityscapes is the
  internal domain every teacher knows. The distillation pool is fixed to
  `{cs, bdd, idd}`, and each teacher is blind to one domain (external data).
* Custom: `protocol.stages: [{name, teacher, distill_on, external?, epochs?}]`
  with `teachers.per_teacher.<name>.train_on`, or `checkpoint` to bring your
  own teacher.

## Outputs (`output_dir`)

| File | Contents |
|---|---|
| `eda/` | Per-domain/split stats (sizes, class pixel share and presence, ignore %, RGB moments, raw label values), example GT overlays, cross-domain class and brightness plots |
| `results.md`, `results.json`, `miou_matrix.csv` | R[stage, domain] mIoU, final average, forgetting, BWT, teacher upper bounds |
| `miou_matrix.png`, `forgetting_curve.png`, `loss_curves.png` | Plots |
| `predictions_<domain>.png` | Image, ground truth and student prediction |
| `diagnostics.json` | Router external score per domain, prototype observations and rejected outliers, buffer contents |
| `last.pt`, `student_stage_XX.pt` | Resumable state (student, old student, banks, buffer) |
| `resolved_config.yaml`, `manifests/` | Exact settings and split membership |

## Status

The implementation is complete and tested end to end on synthetic data (see the
METHOD.md smoke table). It has **not** been run on Cityscapes, BDD or IDD yet,
so no claims are made about real-data performance.
