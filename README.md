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

## Per-epoch report

Every epoch, both for the teachers and for each distillation stage, prints:

```
[bdd100k] ep[3/10]: train loss 1.9510 acc 86.32 (teacher-agree 49.52)
    val   loss 0.8228 acc 87.71 | mIoU 40.98 | mF1 45.00 | bIoU 3.78
    loss terms: kd_current=1.22 kd_replay=0.22 bn_domain=0.26 bn_layer=0.21 class_proto=0.04
    val mIoU per seen domain: cityscapes=43.83, bdd100k=40.98 | avg 42.40
    forgetting so far: avg 2.37 | cityscapes=2.37
    BWT so far: avg -1.43 | cityscapes=-1.43
    router external score: bdd100k=0.052
    buffer: 40 exemplars over 1 stage(s)
    lr / time: 2.23e-04 / 95s
    class            IoU      F1    bIoU
    road           86.75   92.90    7.54
    ...
```

* **Train accuracy** is measured against ground truth where it exists. It is a
  measurement only: the student does not learn from labels unless
  `supervised_ce` is on.
* **Teacher-agree** is the fraction of pixels where the student matches the
  teacher's prediction.
* **Validation** uses the held-out *val* split, never the test split. The
  headline line and the per-class table cover the current stage's domains.
* **Per seen domain, forgetting and BWT** cover every domain learned so far.
  Forgetting is the best earlier val mIoU minus the current one. BWT is the
  current val mIoU minus the value at the end of the stage where the domain was
  learned.
* **Boundary IoU** follows Cheng et al. (2021), with a band width of
  `metrics.boundary_ratio` × the image diagonal.
* `eval_every`, `metrics.print_per_class` and `eval.max_batches_epoch` control
  how often validation runs and how much it costs.

## Outputs (`output_dir`)

| File | Contents |
|---|---|
| `eda/` | Per-domain/split stats (sizes, class pixel share and presence, ignore %, RGB moments, raw label values), example GT overlays, cross-domain class and brightness plots |
| `results.md`, `results.json`, `miou_matrix.csv` | R[stage, domain] mIoU, final average, forgetting, BWT, teacher upper bounds |
| `miou_matrix.png`, `forgetting_curve.png`, `loss_curves.png`, `training_curves.png` | Plots. `training_curves.png` shows train/val loss and accuracy, val mIoU/bIoU/F1, seen-average mIoU, forgetting and BWT per epoch |
| `history.csv`, `per_class_val_history.csv` | Every per-epoch metric, plus per-class IoU/F1/bIoU per epoch |
| `predictions_<domain>.png` | Image, ground truth and student prediction |
| `diagnostics.json` | Router external score per domain, prototype observations and rejected outliers, buffer contents |
| `last.pt`, `student_stage_XX.pt` | Resumable state (student, old student, banks, buffer) |
| `resolved_config.yaml`, `manifests/` | Exact settings and split membership |

## Status

The implementation is complete and tested end to end on synthetic data (see the
METHOD.md smoke table). It has **not** been run on Cityscapes, BDD or IDD yet,
so no claims are made about real-data performance.
