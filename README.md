# SemiSL_Consistency_FixMatch

**FixMatch semi-supervised classification on Crop36 (FashionLetters36) - PAI-SG SemiSL**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the semi-supervised
comparison in [SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36).

## Method

Cross-entropy of the strong view (RandAugment 2 ops + Cutout) against the argmax of the weak view where max prob >= 0.95; unlabeled ratio 7. [Unified] the weak unlabeled view has its own no-grad forward (BN statistics not shared) so that 64 + 448 + 448 images fit in 8 GB.

**Common setup** (identical for the four classification repositories; all methods side by side:
[SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36)). Crop36 (FashionLetters36 object crops, 36 classes): labeled = the crops of the labeled images of the
nested image splits (seed 777; 1 % = 41 images / 258 crops, 5 % = 205 / 1,340, 10 % = 410 / 2,653; lists in
`splits/`), unlabeled = all 26,716 train crops with labels never read, test = 6,699 crops. ResNet-18 from scratch or
from the VICReg 200-epoch SSL backbone ([SSL_Redundancy_VICReg](https://github.com/Statera-Workshop/SSL_Redundancy_VICReg)). Microsoft USB
classic_cv defaults: SGD lr 0.03 nesterov, wd 5e-4 (not on BN / bias), labeled batch 64, cosine 7/16, EMA 0.999
model evaluated, 10k steps, last step reported. GPU augmentation; no horizontal flip. Supervised-only reference
(same recipe): 0.502 / 0.833 / 0.873 from scratch,
0.780 / 0.892 / 0.908 from VICReg (1 / 5 / 10 %); 100 % labels 0.926 / 0.932.

## Results (Crop36 EMA test top-1)

| Init | 1 % | 5 % | 10 % |
| --- | --- | --- | --- |
| from scratch | 0.860 | 0.911 | 0.919 |
| VICReg 200 | 0.797 | 0.908 | 0.916 |
| supervised only, from scratch | 0.502 | 0.833 | 0.873 |
| supervised only, VICReg 200 | 0.780 | 0.892 | 0.908 |

Within 1 point of full supervision with 5-10 % labels; 0.860 with 1 % labels (+0.36 over supervised-only). At 1 % the SSL initialisation hurts (0.860 -> 0.797) - presumably confident wrong pseudo-labels early (pseudo-label accuracy was not logged).
Macro F1 and the evaluation curve of every run: `results/`.

## Run

```bash
python code/make_label_splits.py           # nested labeled splits (needs FL36 at ~/SSL_Wave1/datasets)
python code/semisl_cls.py --method fixmatch --ratio p10 --init scratch --tag cls_p10_fixmatch_scratch
python code/semisl_cls.py --method fixmatch --ratio p10 --init <ssl_vicreg_e200_backbone.pth> --tag cls_p10_fixmatch_vicreg200
```

Crop36 is built by `prep_ssl_data.py` of [SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36). `code/semisl_cls.py` holds all five methods; this repository's
runs use `--method fixmatch` (the supervised-only runs are included as the baseline).
Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md) (USB MIT notice in `licenses/`).
