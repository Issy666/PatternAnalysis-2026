# ConvNeXt for Alzheimer's Disease Classification on ADNI Brain MRI

**COMP3710 Project 2.1 (Classification) · ConvNeXt · Hard difficulty** · Isabel Barton

Summary: A PyTorch ConvNeXt that classifies T1-weighted structural brain MRI slices from the ADNI dataset as **Alzheimer's disease (AD)** or **cognitively normal (NC)**. The final classification combines each scan's 20 MRI slices into one classification, and studies the question: *does the convnext model know when it is unsure or overconfident?*

The final ConvNeXt- Tiny model reaches **80.5% patient-level test accuracy**, with AD recall above NC recall, calibrated confidence and a referral rule that sends ~17% of scans to a specialist to verify (scans the model is not confident in). A plain CNN with 70x less paramters as a baseline performs as well on the same split with 82.7% test accuracy with no dataset/paramater fine tuning.

Therefore, the main engineering recommendation is that ConvNeXt's extra cost is not justified for this data.

---

## 1. Problem and Algorithm

### 1.1 The problem

Alzheimer's disease causes irreversible neurodegeneration that shows on structural MRI's as hippocampal and cortical atrophy and enlarged ventricles. An automated classifier that labels T1-weighted scans as AD or NC could support screening, but only if it is accurate on *new* patients (e.g. a test set) and, in if its confidence can be trusted (e.g. a confident missed AD case is far worse than a false alarm). The difficulties are natural variation in head shape and age between people, subtle differences between early AD and normal ageing.

The task is binary classification of 2D sagittal slices (256 × 240 greyscale, 20 consecutive slices per scan) from a ADNI data subset [3], with a target of **≥ 0.80 test accuracy**.

### 1.2 The algorithm: ConvNeXt

ConvNeXt [1] is a pure convolutional network which has been modified from a ResNet to mimic similar properties from vision Transformers. Vision Transformers are often seen as the modern state of the art in computer vision. However, ConvNeXt-T matched Swin-T on the ImageNet dataset (82.1% vs 81.3%) at similar compute in paper [1]. ConvNeXt mimics vision Transformers in the following ways:

- **Stage compute ratio:** blocks per stage changed from ResNet's (3, 4, 6, 3) to (3, 3, 9, 3), matching Swin's 1:1:3:1 ratio.
- **Patchify stem:** a 4×4, stride-4 non-overlapping convolution replaces ResNet's 7×7 conv and max-pool, like the patch embedding of ViT/Swin.
- **Inverted bottleneck:** each block expands to 4× the channels and projects back (C -> 4C -> C), like the Transformer MLP block.
- **Large kernels:** the depthwise convolution is moved to the start of the block (as self-attention comes before the MLP) and enlarged to 7×7, matching Swin's 7×7 attention windows.
- **Fewer activations and normalisations:** GELU replaces ReLU, with only one activation and one normalisation per block, as in a Transformer block.

My ConvNeXt is based on the model in [1], and the code was cloned from and then modified from the official implementation [2]. The model as developed in `modules.py` is depicted in Figure 1 below.

![ConvNeXt-T architecture as used in this project](images/convnext_architecture.png)

*Figure 1. ConvNeXt-T as used here (`modules.py`). Each block (below) keeps the shape; downsampling layers halve the resolution and double the channels.*

![A ConvNeXt block](images/convnext_block.png)

*Figure 2. A ConvNeXt block: a large-kernel depthwise convolution mixes space, an inverted-bottleneck (1×1 convs, expansion ×4, one GELU) mixes channels, and the residual branch is scaled by a learnable per-channel value and randomly dropped during training.*

**How it works.** Each MRI slice is padded from 256 x 244 to 256 × 256 and the rgb channels are instead two neighbouring slices as the three input channels. The **patchify stem** turns non-overlapping 4 × 4 pixel patches into 96-channel features. Four **stages** of ConvNeXt blocks (3, 3, 9, 3 blocks) build a feature hierarchy, from edges and boundaries in stage 1 to more complex structures in stages 3-4, while strided 2 × 2 convolutions halve the resolution between stages (64 -> 32 -> 16 -> 8). Global average pooling gives one 768-dimensional vector per slice, and a linear head outputs the logits for AD and NC. Softmax then outputs the classification perr slice proberbility. A scan's diagnosis is the mean P(AD) over its 20 slices, optionally scaled and compared with a threshold, and scans whose probability lies close to the threshold are referred to a specialist.

**Changes from the paper and original git repo**:

| Component | ConvNeXt paper [1] | This project | Why |
|---|---|---|---|
| Normalisation | LayerNorm | **BatchNorm** [15] (`convnext_tiny_bn`); LayerNorm kept as `convnext_tiny` | BatchNorm trained from scratch faster and more accurately (Section 5.1),. Paer [1] also only had  a LayerNorm gain of +0.1% on ImageNet|
| Classifier head | LayerNorm -> linear | BatchNorm1d -> **dropout 0.4** -> linear | extra regularisation against early overfitting |
| Input | RGB image | slices n−1, n, n+1 as the 3 channels (`dataset.NeighbourSliceFolder`) | atrophy is a 3D change; the paper's 3 channels would otherwise hold 3 copies of one slice. THis change resulted in +3.9 accuracy points (Section 5.1) |
| Initialisation | Pretrained ImageNet training, 300 epochs | **from scratch**, 18 epochs | |
| Sizes | Tiny and up | Tiny (27.8 M paramaters) and **Nano** (15.0 M Paramaters) | size benefit comparison |
| Augmentation | RandAugment, Mixup, CutMix, flips | small rotations, intensity jitter, random erasing | keep the anatomy and label of each slice intact (Section 3.4) |

The implementation follows the official ConvNeXt code [2] (blocks, drop path, layer scale, AdamW parameter groups, schedules), adapted for ADNI with reference to other ADNI ConvNeXt code [4], [6]; sources are also cited in the code.

The **baseline** (`simple_cnn`) is a plain CNN with: four blocks of 3 × 3 conv -> BatchNorm -> ReLU -> 2 × 2 max-pool (32–256 channels), global average pooling, dropout 0.4 and a linear head equating to 0.39 M parameters.

---

## Feasibility Review

*Project 2.1 · ConvNeXt on ADNI (AD vs CN) · Hard difficulty*

### 1. User Need, Scope & Acceptance Criteria
**Users:** clinicians (screening aid) and Alzheimer's / medical-imaging ML researchers. **Purpose:** classify 2D T1-weighted brain MRI slices as AD or CN and route low-confidence cases to a specialist. **Open dilemma:** does the model know when it is unsure, or stay overconfident when wrong, compared with a baseline CNN?

| # | Acceptance criterion | Test |
|---|---|---|
| 1 | Test accuracy ≥ 0.80 | ADNI test split, evaluated once after selection on validation |
| 2 | Missing AD is worse than a false alarm | AD recall ≥ CN recall (confusion matrix); also precision, macro-F1, AUROC |
| 3 | Confidence tracks correctness better than the baseline | Reliability diagram and ECE vs baseline CNN; a reject rule routing ≤ 20% of cases to a human raises accuracy on the rest |
| 4 | Uses anatomy, not shortcuts | Grad-CAM focuses on the brain, not the background or border |
| 5 | Resource: fits one RTX 3080 (10 GB) | Peak training VRAM < 10 GB; one run < 12 h; latency and parameters vs baseline |

### 2. Model Choice & Course Concepts
- **Data:** course ADNI subset, 1,526 patients × 20 sagittal 256×240 slices, ~48% AD. Split by patient: train 915, validation 161 (15%), test 450 (used once). Greyscale -> 3 channels, padded to 256², training-set mean/std normalisation.
- **Architecture:** ConvNeXt-T [1] (blocks 3-3-9-3, channels 96–768, 27.8 M parameters): 4×4 patchify stem; blocks of 7×7 depthwise conv -> LayerNorm -> 1×1 conv ×4 -> GELU -> 1×1 conv, with layer scale, drop path and residual; pooled -> 2-way linear head, softmax = confidence. Paper training settings where possible (AdamW, wd 0.05, warmup + cosine LR, label smoothing 0.1), batch 4096 -> 64 with LR scaled.
- **Course concepts:** convolution's inductive biases (locality, translation equivariance) fit atrophy, which appears as local shapes at shifting positions, and need less data than a ViT for ~1,500 patients. Strided downsampling builds hierarchical features (edges -> ventricles, cortex); stochastic depth, weight decay and normalisation limit overfitting.
- **Literature:** ConvNeXt-T beats Swin-T on ImageNet (82.1% vs 81.3%) as a pure CNN [1], so 0.80 is plausible. Mixup/CutMix blend two brains, so they will be tested on and off.

### 3. Preliminary Feasibility Evidence
**Data audit:** 20 slices per patient, all 256×240, no patient in more than one split. **Smoke test:** official ConvNeXt code [2] with an ADNI loader, ImageNet-pretrained ConvNeXt-T fine-tuned on CPU (128², 11 epochs, batch 64, LR 1e-4).

| Measurement | Result | Measurement | Result |
|---|---|---|---|
| Training time (CPU) | 4 h 52 min | Test accuracy (9,000 slices) | **75.6%** |
| Best validation accuracy | 92.8% (epoch 9) | AD / CN recall (estimated) | 57% / 94% |
| Parameters | 27.8 M | CPU inference (128²) | 39 ms per slice |

The pipeline works end to end (validation loss 0.42 -> 0.20). Test accuracy is below target with a large validation–test gap, and most errors are missed AD (against the acceptence criteria 2). Grad-CAM often highlights the ventricles and medial temporal lobe, but sometimes the border (possible shortcut).

### 4. Risks, Budget & Fallback
- **Risks -> mitigations:** 
*long training* (CPU ~5 h for 11 epochs) -> GPU with mixed precision, short runs first, epochs tuned on validation (50–100). 
*Below 0.80, biased against AD* -> 256² input, patient-level averaging of 20 slices, validation-chosen threshold favouring AD recall. 
*Validation unlike test* -> audit the split, select on validation only. 
*No Swin-T run* -> cite the paper and complete the baseline CNN as the required comparison.
- **Budget:** RTX 3080 (10 GB, 24/7) for all development and benchmarks so numbers are comparable; lab GPUs and Rangpur A100 as backup.
- **Next experiment:** my own ConvNeXt-T trained from scratch on the RTX 3080, compared with the pretrained smoke test and tuned towards 0.80 test accuracy; then the baseline CNN and calibration analysis.
- **Fallback:** Project 2.2, 2D U-Net brain segmentation on OASIS (Easy, capped at 10/20): same convolution concepts and brain MRI, and similar to Demo 2.

---

## 3. Data, Pre-processing and Splits

### 3.1 Dataset

The course's ADNI subset [3] contains 30,520 JPEG slices: **1,526 scans** (743 AD, 783 NC) with 20 consecutive sagittal slices each, 256 × 240 greyscale. Files are named `<id>_<slice>.jpeg`. 

### 3.2 Patient leakage across ADNI visits

**The problem.** The `<id>` in each filename is the ADNI **image ID of one scan session**, not the person. `meta_data_with_label.json` (supplied with the dataset) is keyed by the same IDs (all 1,525 match, and its labels agree with the `AD`/`NC` folders for every scan), and each entry gives the original ADNI file name, which contains the subject ID, sequence and acquisition date:

```
ADNI_068_S_0473_MR_Accelerated_Sagittal_MPRAGE__br_raw_20171116173455030_..._I932015.nii
     └ subject       └ sequence                         └ 2017-11-16          └ image ID (= filename ID)
```

The 1,525 scans come from only **680 people** (221 AD, 459 NC), many scanned at several visits (up to 8 scans). The course's train/test split and any split by filename ID therefore put the same people in more than one split: **216 people** are in both the course training folder and test, and 95% of the AD scans in a filename-ID validation split (74 of 78) belonged to people also in training.

On NC, people seen in training scored 0.99 and new people 0.88. The model largely recognising **people** rather than disease.

### 3.3 Leakage-free subject-level split

`subject_datasplit.py` re-splits all 1,525 scans **by ADNI subject**, so no person appears in more than one split. The final train/validate/test split 60/ 10 / 30% was:

| Split | Class | Scans | Slices |
|---|---|---|---|
| train | AD | 443 | 8,860 | 
| train | NC | 470 | 9,400 |
| validation | AD | 72 | 1,440 |
| validation | NC | 78 | 1,560 | 
| test | AD | 227 | 4,540 | 68 | 
| test | NC | 235 | 4,700 | 142 | 

**What it changed** (LayerNorm ConvNeXt-T with identical settings on different dataset splits):

| | Initial Course split | New Patient split |
|---|---|---|
| Best validation slice accuracy | 84.8% | 75.7% |
| Test slice accuracy | 68.2% | 69.6% |
| **Validation − test gap** | **16.6 points** | **6.1 points** |
| Test patient AD / NC recall | **0.42 / 0.96** | **0.75 / 0.69** |

Removing the leak made validation more honest and the gap shrank from 17 to 6 points and removed the strong NC bias. All results below use this split.

### 3.4 Pre-processing and augmentation

All transforms are in `normalise.py`; the dataset classes are in `dataset.py`.

| Step | Train | Validation / test | Reason |
|---|---|---|---|
| Pad to square (`PadToSquare`) | Y | Y | 256 × 240 -> 256 × 256 with black rows; this way the brain is never stretched |
| 3 channels | Y | Y | neighbouring slices n−1, n, n+1 of the same scan as R, G, B; edge slices repeat themselves; no data crosses splits |
| Random alterations | ±15°, shift ±8%, zoom ±10% | N | head tilt, position and size differences between people (age, genser) and scanners |
| Brightness / contrast | ±30% | N | scanner and acquisition intensity differences |
| Normalise | Y | Y | mean 0.1085, std 0.2197 computed on the **training split only** and cached in `stats/` |
| Random erasing [20] | p = 0.25, one noise-filled rectangle | N | occlusion robustness, as in ConvNeXt [1] |

**Deliberately not used.** Horizontal flips (all sagittal slices face the same way; a mirrored brain never appears at test), and **Mixup/CutMix** [1], which blend or paste two brains so the label no longer matches the anatomy, which is what the features should be detecting / learning.

---

## 4. Training

`main.py` trains with `train.py` (training, validation, testing and checkpointing), validating every epoch and testing the best validation checkpoint once at the end. `testscripts/run_pipeline.sh` wraps the whole run.

| Setting | Final value | Reason |
|---|---|---|
| Optimiser | AdamW [18] with weight decay 0.05 | ConvNeXt's optimiser and decay [1]. Weight decay shrinks weights towards 0 each step to reduce overfitting. |
| Learning rate | 5e-4 peak, 3-epoch linear warmup, cosine decay to 1e-6 | initial high learning rate with peaks and then decays overtime as exploration becomes less important |
| Epochs | **18** | validation loss is converges at around epoch 10 in every run, so the cosine decay is made to finish there |
| Batch size | 64 (285 steps/epoch) | fits the RTX 3080; stable BatchNorm statistics |
| Loss | cross-entropy, label smoothing 0.1 [10] | paper's setting; discourages over-confidence |
| Regularisation | drop path 0.3, head dropout 0.4, augmentation (3.4) | the 28 M-parameter model overfits ~400 people quickly so this attempts to reduce overftting|
| Checkpoint selection | best **patient-level validation macro-F1** |  |

![Training and validation curves of the final model](images/training_curves_final.png)

*Figure 4. Final model: validation loss is lowest at epoch 10 (0.495) and rises slightly afterwards (0.544) while training loss keeps falling (0.446): the beginning of overfitting, which is why runs are short.*

![Learning rate schedule](images/lr_schedule.png)

*Figure 5. Learning rate per epoch (mean over each epoch's 285 steps): 3 warmup epochs, then cosine decay.*

**Why not train longer?** In every run (18 or 30 epochs) validation loss reached its minimum around epochs 10-13 and then rose while training loss kept falling. 

---

## 5. Results

All numbers are on the **subject-level test set** (462 scans from 210 people) unless stated. "Patient" means one scan (the mean of its 20 slices).

### 5.1 Development: what each change contributed

| Run | Model | Changes | Test patient acc. | Patient AUROC | AD / NC recall |
|---|---|---|---|---|---|
| `ln_subjsplit` | ConvNeXt-T, LayerNorm | 30 epochs, single-slice input | 72.1% | 0.814 | 0.75 / 0.69 |
| `bn_subjsplit` | ConvNeXt-T, **BatchNorm** + head dropout | as above | 73.8% | 0.840 | 0.88 / 0.60 |
| `bn_subjsplit_25d` | ConvNeXt-T, BN | extra dimensions via neighbouring channels | 77.7% | 0.870 | 0.90 / 0.66 |
| `bn_25d_aug_e18` | ConvNeXt-T, BN | **+ stronger augmentation, 18 epochs** | **80.5%** | **0.893** | 0.92 / 0.69 |
| `bn_25d_aug_e18_step2` | ConvNeXt-T, BN | neighbours 2 slices apart | 76.6% | 0.877 | 0.91 / 0.63 |
| `bn_nano_25d_aug_e18` | ConvNeXt-**Nano**, BN | smaller model | 79.2% | 0.887 | 0.89 / 0.70 |

- **Extra dimensions via neighbouring channels** was the clearest gain: +3.9 points patient accuracy, at no extra cost.
- **BatchNorm vs LayerNorm**: +1.7 points with single-slice input. LayerNorm learned slowly (validation near 50–60% for the first ~5 epochs).
- **Stronger augmentation & 18 epochs**: +2.8 points and halved the training time.
- **Nano vs Tiny**: −1.3 points.

### 5.2 Final model

The final model is **BatchNorm ConvNeXt-T with extra dimensions via neighbouring channels on input and the 18-epochs**. 

3 tests with the final model:

| Seed | Slice acc. | Slice AUROC | **Patient acc.** | Patient AUROC | AD / NC recall | Macro-F1 |
|---|---|---|---|---|---|---|
| 0 | 76.1% | 0.842 | 80.5% | 0.893 | 0.92 / 0.69 | 0.803 |
| 1 | 74.1% | 0.827 | 80.5% | 0.871 | 0.82 / 0.79 | 0.805 |
| 2 | 74.6% | 0.825 | 77.3% | 0.875 | 0.78 / 0.76 | 0.773 |
| **Mean** | **74.9%** | **0.831** | **79.4%** (77.3–80.5) | **0.880** | **0.84 / 0.75** | **0.794** |

![Confusion matrices, seed 0](images/confusion_matrix.png)

*Figure 6. Seed 0 at threshold 0.5. Averaging the 20 slices of a scan halves the missed AD (16.5% of slices -> 7.9% of scans); false alarms stay at ~31% because CN people the model calls AD have most of their slices called AD.*

![ROC curves, seed 0](images/roc_curve.png)

*Figure 7. ROC curves (AD = positive). Scan-level AUROC 0.893 exceeds slice-level 0.842 almost everywhere; the dots mark threshold 0.5.*

**Acceptance criteria from the feasibility review:**

| # | Criterion | Result | Met? |
|---|---|---|---|
| 1 | Test accuracy ≥ 0.80 | 79.4% patient-level (≥ 0.80 in 2 of 3 seeds); slice-level 74.9% | Y |
| 2 | AD recall ≥ NC recall | 0.92/0.69, 0.82/0.79, 0.78/0.76 | Y |
| 3 | Confidence tracks correctness better than the baseline; ≤ 20% referral raises accuracy | referral: 16.7% of scans -> 77.9% to 83.4% | Y |
| 4 | Grad-CAM on anatomy, not shortcuts | heat maps favour the brain | Partly |
| 5 | Fits one RTX 3080: VRAM < 10 GB, run < 12 h | 5.2 GB, 15.5 min per run | Y |

---

## 6. Open Research Dilemma

*Does the model reliably quantify uncertainty on ambiguous cases compared to a simple baseline, or does it stay overconfident when wrong? What confidence threshold or reject rule safely automates confident cases and routes the rest to specialists, and does ConvNeXt's extra cost justify itself?*

Everything below compares ConvNeXt-T with the baseline CNN trained with the identical data split, 3 channel dimensions, augmentation, optimiser, schedule and checkpoint rule, only the architecture differs. 

### 6.1 Quantitative benchmarking

| Test, patient level | Baseline CNN | ConvNeXt-T (seed 0) | ConvNeXt-T (3-seed mean) | ConvNeXt-Nano |
|---|---|---|---|---|
| Accuracy | **82.7%** | 80.5% | 79.4% | 79.2% |
| AUROC | **0.904** | 0.893 | 0.880 | 0.887 |
| AD recall / NC recall | 0.81 / **0.85** | **0.92** / 0.69 | 0.84 / 0.75 | 0.89 / 0.70 |
| Macro-F1 | **0.827** | 0.803 | 0.794 | 0.791 |
| Slice accuracy / AUROC | **79.3% / 0.875** | 76.1% / 0.842 | 74.9% / 0.831 | 75.5% / 0.834 |

ConvNeXt-T minus baseline: accuracy −2.2 points, AUROC −0.011. ConvNeXt leans more towards AD (higher AD recall, more false alarms). Both models achieve similar results. 

### 6.2 Calibration and the reject option

Confidence is the probability of the predicted class. 

Calibration is measured with the expected calibration error (ECE, 10 bins over [0.5, 1]) and reliability diagrams. 

**Temperature scaling** [12] divides the logits by a value T fitted on the **validation** set minimising the spread. It never changes a slice's predicted class. 

The decision threshold and a **referral band** (refer a scan if its P(AD) is within ±b of the threshold, with b set so that 20% of validation scans are referred, then everything is applied once to test (`calibrate.py`).

| Test, patient level | Baseline CNN | ConvNeXt-T seed 0 | ConvNeXt-T 3 seeds | ConvNeXt-Nano | ConvNeXt-T LayerNorm |
|---|---|---|---|---|---|
| Fitted temperature T | 0.59 | 0.69 | 0.51–0.69 | 0.35 | 0.81 |
| ECE raw -> calibrated | 0.096 -> **0.040** | 0.085 -> 0.057 | 0.098 -> 0.050 (mean) | 0.107 -> 0.044 | 0.057 -> 0.053 |
| Mean confidence when correct / wrong | 0.76 / 0.64 | 0.77 / 0.65 | 0.70–0.77 / 0.60–0.65 | 0.72 / 0.62 | 0.74 / 0.67 |
| Refer least-confident 20% -> accuracy of the rest | 82.7 -> **89.5%** | 80.5 -> 87.8% | -> 84.6–87.8% | 79.2 -> 85.7% | 74.9 -> 78.4% |


![Reliability diagrams for ConvNeXt before and after temperature scaling](images/reliability_convnext.png)

*Figure 8. ConvNeXt-T seed 0, test scans: before (left) and after (right) temperature scaling fitted on validation.*

![Reliability diagrams for the baseline](images/reliability_baseline.png)

*Figure 9. The baseline CNN, same procedure.*

![Reject curve](images/reject_curve.png)

*Figure 10. ConvNeXt-T seed 0: accuracy of the scans the model keeps as the least-confident ones are referred, against an oracle that refers the wrong ones first and random referral (flat).*

**Findings.**
1. **Both models are under-confident at scan level** before calibration (T < 1 for every model): averaging 20 slices pulls each scan's probability towards 0.5. Temperature scaling fitted on validation roughly halves the patient-level ECE and transfers to test, at the price of worse slice-level calibration.
2. **Confidence is informative but not decisive.** Correct predictions are more confident than wrong ones by only ~0.12, and referring the least-confident 20% raises accuracy by 7 points for both models; but half of all errors are still made confidently. The model "partly knows" when it is unsure.
3. **ConvNeXt is not better at knowing when it is unsure than the baseline**: the baseline has the lowest calibrated ECE, the largest confidence gap and the best reject curve. 


### 6.3 Computational efficiency

Measured with one RTX 3080:

| Model | Parameters | Weights | Peak training VRAM | Training time (18 epochs) | GPU latency / scan | GPU throughput | CPU latency / scan |
|---|---|---|---|---|---|---|---|
| Baseline CNN | **0.39 M** | **1.5 MB** | **1.9 GB** | **4 min** | **2.9 ms** | **7,200 slices/s** | **142 ms** |
| ConvNeXt-Nano (BN) | 15.0 M | 57 MB | 3.3 GB | 10 min | 9.9 ms | 2,100 slices/s | 321 ms |
| ConvNeXt-T (BN) | 27.8 M | 106 MB | 5.2 GB | 15.5 min | 16.3 ms | 1,300 slices/s | 648 ms |
| ConvNeXt-T (LayerNorm) | 27.8 M | 106 MB | 5.7 GB | 18.5 min | 18.9 ms | 1,100 slices/s | 537 ms |

All models comfortably meet the resource criterion, but ConvNeXt-T costs **~70x the parameters, ~4x the training time and ~5.6x the GPU latency** of the baseline for no measurable gain in accuracy or calibration.

### 6.4 Qualitative error analysis

![Failure cases with Grad-CAM and per-slice P(AD)](images/failure_cases.png)

*Figure 11. Four representative failures of ConvNeXt-T seed 0 (left: middle slice with Grad-CAM [13] for the predicted class, stage 3; right: P(AD) of each of the scan's 20 slices, dashed line = 0.5, solid line = scan mean). Made by `testscripts/report_figures.py`; the case details and person-level statistics below come from `testscripts/compare_runs.py RUN --failures`.*

| Case | Scan (subject, year) | Truth -> prediction | What happened | Root cause (diagnosis) |
|---|---|---|---|---|
| **A. Confident missed AD** | `1005735` (`003_S_6264`, 2018) | AD -> NC, P(AD) 0.14 | every slice below 0.31; the same person's 2019 scan is also missed (0.27) | **person-level appearance**: this person's brain looks NC-like to the model on every slice and visit, consistent with mild or early-stage atrophy. Grad-CAM sits on the ventricles/corpus callosum, i.e. plausible anatomy that simply did not look abnormal |
| **B. Confident false alarm** | `959645` (`041_S_6192`, 2018) | NC -> AD, P(AD) 0.87 | 18 of 20 slices called AD; the second scan from the same session (`959644`) gives the same 0.87 | Grad-CAM peaks at the **posterior edge of the brain/cerebellum and partly outside it**, the same hot spot as correct AD predictions (Figure 13): a possible **shortcut on the skull-stripping boundary** rather than atrophy |
| **C. Repeated false alarm** | `1070417` (`068_S_0127`, 2018) | NC -> AD, P(AD) 0.86 | all 3 of this person's test scans are false alarms | **person-level**: a long-followed ADNI participant (enrolled in ADNI-1) whose anatomy looks AD-like, plausibly age-related atrophy; age is not in the metadata, so this cannot be confirmed. Across the test set, 18 of 71 NC people with several scans have *every* scan flagged |
| **D. Borderline, correctly referred** | `301063` (`126_S_4494`, 2012) | AD -> NC, P(AD) 0.50 | central slices 0.75–0.90 (AD), outer slices 0.05–0.45 (NC); the mean lands on the boundary | **slice disagreement**: the evidence is in the central slices only; the scan-level mean dilutes it. The referral rule correctly sends this case to a specialist |
| **E. Out-of-distribution input** (course split) | `985197` | AD -> NC, ~0.95 confident | corrupted orientation and intensity on all slices (Figure 3) | **data quality**: softmax confidence did not flag an input unlike anything in training; removed from the data (Section 3.1) |

![Grad-CAM examples per outcome](images/gradcam_examples.png)

*Figure 12. Grad-CAM (stage 3, predicted class) on the middle slice of test scans per outcome.*

![Mean Grad-CAM per outcome](images/gradcam_mean.png)

*Figure 13. Mean Grad-CAM per outcome. Correct AD and false alarms share the same posterior hot spot; correct NC predictions focus more centrally (ventricles, corpus callosum).*

**Grad-CAM statistics** (`visualise.py`): on average 47% of each heat map lies inside the (filled) brain outline, which covers 24% of the image, a 1.9× enrichment over an evenly spread map; no scan has ≥ 75% inside. Part of the "outside" mass is the 16 × 16-pixel resolution of stage-3 maps bleeding over the brain edge, but the shared posterior hot spot of correct and false AD calls suggests the model also uses the brain's outline. Referred scans have the least focused maps (37% inside).

![UMAP of scan-level features](images/umap_scans.png)

*Figure 14. UMAP [14] of the pooled features (one point per scan). False alarms sit **inside** the AD cluster (confident errors), while referred scans and missed AD lie where the two clusters meet.*

![UMAP of slice-level features](images/umap_slices.png)

*Figure 15. The same features per slice: a continuous horseshoe from AD-like to NC-like slices with a mixed band where most slice errors fall; averaging 20 slices per scan sharpens the separation (slice accuracy 76.1% -> scan accuracy 80.5%).*

**Summary of failure modes.** Errors are **person-level** rather than random: a person whose anatomy the model finds AD-like (or NC-like) is misclassified on every slice and every visit, and these errors are confident, so they escape the referral rule. Slice-level disagreement causes the borderline cases, which the referral rule does catch. Out-of-distribution inputs are not flagged by softmax confidence at all.

### 6.5 Engineering recommendation

For a project manager deciding whether to deploy this as a triage assistant:

1. **Use the plain CNN, not ConvNeXt, on data of this size.** The 0.39 M-parameter baseline matched ConvNeXt-T in accuracy (82.7% vs 79.4%, not significantly different), calibration and referral performance, at 1/70 of the parameters, ~1/4 of the training time and ~1/6 of the latency. ConvNeXt's ImageNet-scale design choices do not pay off when training from scratch on ~400 people; if ConvNeXt is required, use Nano (same accuracy, half the cost of Tiny).
2. **Decide per scan, not per slice**, by averaging the 20 slices: +3–6 points accuracy and higher AUROC for every model.
3. **Calibrate with temperature scaling on held-out data** before trusting the confidence scores (patient ECE roughly halves); keep the decision threshold at 0.5 unless it can be tuned on a much larger validation set (or by cross-validation).
4. **Operating rule:** automate scans whose calibrated P(AD) is outside ±0.13 of the threshold and refer the rest. On test this referred **16.7%** of ConvNeXt's scans and raised accuracy on the automated 83% from 77.9% to **83.4%**, with **AD recall 0.96** on the kept scans; the baseline under the same rule: 13.9% referred, 87.4% kept accuracy.
5. **Do not rely on confidence alone for safety.** About half of the errors are confident and person-specific (cases A–C), and corrupted inputs are not flagged (case E). Add an input-quality / out-of-distribution check and a second reader for screen-positive cases.
6. **Split by subject, never by scan or filename.** The leakage found here inflated validation accuracy by ~17 points and created a strong NC bias; any reported ADNI result should state how repeated visits were handled.

---

## 7. Usage

### 7.1 Files

| File | Purpose |
|---|---|
| `modules.py` | the models: `ConvNeXt` (LayerNorm, as in the paper), `ConvNeXtBN` (BatchNorm + head dropout), `SimpleCNN` (baseline), the `--model` registry and `build_model` |
| `dataset.py` | `build_dataset` (train/val/test `ImageFolder`s with the right transforms), `NeighbourSliceFolder` (2.5D input), `patient_id` |
| `normalise.py` | `PadToSquare`, training-set mean/std (cached in `stats/`), train and eval transforms |
| `subject_datasplit.py` | leakage-free train/validation/test split by ADNI subject (Section 3.3) |
| `train.py` | AdamW optimiser, `train_one_epoch`, `evaluate`, AD/NC metrics |
| `main.py` | training entry point: trains, validates every epoch, saves checkpoints, tests the best checkpoint |
| `predict.py` | runs a trained checkpoint on the test set or analyses saved predictions: patient-level metrics, confusion matrix, ROC, calibration, reject curve, example predictions |
| `calibrate.py` | temperature scaling, validation-chosen threshold and referral rule (Section 6.2) |
| `visualise.py` | Grad-CAM and UMAP (Section 6.4) |
| `utils.py` | logging, AMP loss scaler, learning-rate schedule, checkpoint saving/resuming/loading |
| `../../testscripts/run_pipeline.sh` | one complete run: data check -> training -> test -> analysis -> plots |
| `../../testscripts/check_data.py` | data checks and leakage checks (filename IDs and ADNI subjects), recomputes normalisation statistics |
| `../../testscripts/compare_runs.py` | scan-level summaries, paired bootstrap comparisons between runs or ensembles, failure analysis, seen-vs-new-person breakdown |
| `../../testscripts/benchmark.py` | resource profile: parameters, VRAM, training time, GPU/CPU latency and throughput (Section 6.3) |
| `../../testscripts/report_figures.py` | the example input/output and failure-case figures (Figures 11 and 16) |
| `../../testscripts/plot_log.py` | training curves and learning-rate plot from `log.txt` |
| `../../testscripts/check_scan_era.py` | scan-era confound check (Section 3.2) |

### 7.2 Running

**1. Create the subject-level split** (once). Copy the course's `ADNI/AD_NC/{train,test}` images into one `train/` folder with `AD/` and `NC/` subfolders, next to `meta_data_with_label.json`, then:

```bash
python subject_datasplit.py --root ~/Desktop/data/ADNI_patientleakagefix/AD_NC \
    --meta ~/Desktop/data/ADNI_patientleakagefix/meta_data_with_label.json --val 10 --test 30 --dry-run
# check the summary, then run again without --dry-run to move the files
```

**2. Train and test the final model** (≈ 15 min on an RTX 3080):

```bash
cd ../../testscripts
NAME=bn_25d_aug_e18 nohup ./run_pipeline.sh \
    --model convnext_tiny_bn --neighbour_slices True \
    --epochs 18 --warmup_epochs 3 --drop_path 0.3 \
    --affine_degrees 15 --affine_translate 0.08 --affine_scale 0.1 \
    --brightness 0.3 --contrast 0.3 \
    --best_metric patient_macro_f1 --seed 0 > ~/Desktop/results/bn_25d_aug_e18.log 2>&1 &
tail -f ~/Desktop/results/bn_25d_aug_e18/train.log
```

Swap `--model` for `simple_cnn` (baseline), `convnext_nano_bn` or `convnext_tiny`, and `--seed` for 1 or 2 to reproduce the other runs. Re-running the same `NAME` resumes from the last checkpoint; `QUICK=1 ./run_pipeline.sh` is a one-epoch smoke test.

**3. Calibrate, then visualise:**

```bash
cd ../recognition/ConvNeXt_IsabelBarton
python calibrate.py --resume ~/Desktop/results/bn_25d_aug_e18/checkpoint-best.pth
python visualise.py --resume ~/Desktop/results/bn_25d_aug_e18/checkpoint-best.pth
```

**4. Test a trained model again / analyse predictions** (works on a CPU):

```bash
python predict.py --resume ~/Desktop/results/bn_25d_aug_e18/checkpoint-best.pth       # re-runs the model
python predict.py --predictions ~/Desktop/results/bn_25d_aug_e18/test_predictions.csv  # analysis only, seconds
```

Every script rebuilds the model and its data from the arguments stored in the checkpoint, so nothing about the run has to be repeated on the command line. `python main.py --help` lists all training options (model, learning rate, epochs, augmentation strengths, data paths, 2.5D input, checkpoint metric).

### 7.3 Example inputs and outputs

![Example input and output](images/example_input_output.png)

*Figure 16. Input: three neighbouring slices of a test scan become the R, G and B channels of one padded, normalised 3 × 256 × 256 tensor. Output: P(AD) for each of the scan's 20 slices; their mean (0.85) gives the diagnosis AD (correct). Made by `testscripts/report_figures.py`.*

A run writes to its output folder:

| Output | Contents |
|---|---|
| `log.txt` | one JSON line per epoch: training loss/accuracy/lr, validation loss, accuracy, AD/NC recall and patient-level metrics |
| `test_results.json` | test accuracy, loss, recalls, precisions, macro-F1, confusion counts, best epoch, training time, peak VRAM, parameters |
| `test_predictions.csv` | one row per slice: `path,label,pred,prob_AD,prob_NC` |
| `patient_predictions.csv`, `patient_results.json` | one row per scan (mean P(AD), prediction) and scan-level metrics |
| `confusion_matrix.png`, `roc_curve.png`, `reliability_diagram.png`, `reject_curve.png`, `example_predictions.png` | test-set plots |
| `training_curves.png`, `lr_schedule.png` | training plots |
| `calibrate/`, `visualise/` | calibration results and Grad-CAM / UMAP figures |

Example `test_predictions.csv` rows and the console summary of `predict.py` for the final model:

```
path,label,pred,prob_AD,prob_NC
.../AD_NC/test/AD/1003730_100.jpeg,AD,NC,0.051654,0.948346

Patient level (462 patients, mean prob_AD over slices, threshold 0.5)
  accuracy 80.52%   AD recall 0.921   NC recall 0.694   macro-F1 0.803
  TP 209  FN 18  FP 72  TN 163
```

---

## 8. Dependencies and Reproducibility

### 8.1 Dependencies

Tested with these exact versions on Linux (kernel 7.0), NVIDIA driver 580.178.04, CUDA 13.0, one RTX 3080 (10 GB):

| Package | Version | Used for |
|---|---|---|
| Python | 3.14.8 | |
| PyTorch | 2.14.1+cu130 (cuDNN 9.24) | models, training |
| torchvision | 0.29.1+cu130 | datasets, transforms |
| timm | 1.0.30 | `DropPath`, `trunc_normal_`, label-smoothing loss, random erasing, `accuracy` |
| NumPy | 2.5.4 | statistics, schedules (not used by the model) |
| Pillow | 13.0.0 | image loading |
| matplotlib | 3.11.2 | plots |
| umap-learn | 0.5.12 (with scikit-learn 1.9.1, numba 0.68.0) | UMAP only (`visualise.py`) |

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install timm==1.0.30 numpy==2.5.4 pillow==13.0.0 matplotlib==3.11.2 umap-learn==0.5.12
```

Training needs a CUDA GPU (it always uses mixed precision); `predict.py`, `calibrate.py` and `visualise.py` also run on a CPU.

### 8.2 Reproducibility

- **Data:** the split is deterministic (`subject_datasplit.py --seed 42`) and fully recorded in `split_manifest.csv`; normalisation statistics come from the training split only.
- **Seeds:** `--seed` fixes PyTorch and NumPy; but `cudnn.benchmark` and mixed precision make GPU training non-deterministic, so a rerun with the same seed gives slightly different numbers. The spread between seeds is therefore reported instead of a single run: patient accuracy 77.3–80.5% over seeds 0–2 (Section 5.2).
- **Evaluation:** every analysis script rebuilds the model from the checkpoint's stored arguments, and given the same `test_predictions.csv` all analyses (patient metrics, calibration, ROC, reject curve, Grad-CAM statistics, UMAP with a fixed seed) are exactly reproducible. Re-running a checkpoint reproduces its test probabilities to within 3 × 10⁻⁵ (mixed-precision rounding).
- **Uncertainty:** with 210 test people, accuracy has a margin of about ±5.5 points; model comparisons use paired bootstrap intervals over people.
- **Commands:** Section 7.2 reproduces every reported run (run names match the tables); the comparisons, failure analysis, resource table and the two report-only figures are reproduced by `compare_runs.py`, `benchmark.py` and `report_figures.py` in `testscripts/` (the figures byte-for-byte).

---

## 9. Limitations and Future Work

- **Test-set reuse during development.** Development choices (BatchNorm, 2.5D, augmentation, Tiny vs Nano) were compared on the test set as well as on validation, and Tiny was kept over Nano for its test score. The final numbers are therefore somewhat optimistic. A cleaner protocol is **5-fold cross-validation by subject** on train + validation for every design decision (which also gives a stable threshold and a 5-model ensemble [22]), touching the test set once.
- **Small validation set** (19 AD people): checkpoint selection and the tuned threshold were noisy; the threshold did not transfer.
- **Baseline has one seed**; seeds 1–2 (4 min each) would make the comparison symmetric.
- **Residual confound:** every NC scan is from 2017+, so the scan protocol remains partly predictive of the class; age and disease stage are not available in the metadata, so person-level errors (cases A–C) cannot be attributed to age-related atrophy with certainty.
- **Label smoothing and referral:** label smoothing (0.1, from the paper) is known to degrade selective classification [11]; it was not ablated here, so the reject-rule results may be improvable without it.
- **Not tested:** ImageNet-pretrained initialisation (the strongest lever on small medical datasets [5], [17]), EMA/weight averaging, Mixup/CutMix with a long schedule, coordinate attention [7], other MRI planes (coronal slices performed best in [5]), and an explicit out-of-distribution detector for corrupted scans.
- **Comparability:** the subject-level test set differs from the course test set, so results are not directly comparable with other submissions.

---

## 10. Artificial Intelligence Usage Disclosure

- **Tool used:** Claude (Anthropic), through the Claude Code command-line assistant.
- **What it assisted with:**
  - comparing this implementation with a public reference implementation [4] and with Basereh et al. [5];
  - scaffolding and refactoring code: the BatchNorm ConvNeXt variant, the 2.5D dataset, the smaller ConvNeXt sizes and baseline CNN, the subject-level split script, `calibrate.py` (temperature scaling, threshold and referral rule), `visualise.py` (Grad-CAM, UMAP), the confusion-matrix/ROC/learning-rate plots, and a final code clean-up;
  - analysis scripts: the patient-leakage and scan-era investigation from the ADNI metadata, bootstrap confidence intervals and the resource benchmark;
  - drafting and restructuring this README and condensing the feasibility review.
- **How it was audited and verified:**
  - every experiment was run by me on my own GPU, and every number in this report was read from the run's own output files (`test_results.json`, `log.txt`, `calibration_results.json`);
  - data-handling code was checked directly: subject overlap between splits is re-checked on disk after splitting, `check_data.py` validates every split, and the 2.5D dataset was tested to load the correct neighbouring slice files at scan edges;
  - the code clean-up was verified with a regression test that compared, before and after, the outputs of five trained checkpoints, test-set tensors, the learning-rate schedule, a training step and all analysis outputs (all identical), plus an end-to-end pipeline run and a resume test;
  - statistical claims use paired bootstrap intervals over people, and results that did not hold up (e.g. the validation-tuned threshold, the second neighbour step) are reported rather than dropped.

---

## References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11976–11986. [Online]. Available: https://arxiv.org/abs/2201.03545

[2] Meta AI Research, "ConvNeXt: official PyTorch implementation," GitHub repository. [Online]. Available: https://github.com/facebookresearch/ConvNeXt

[3] Alzheimer's Disease Neuroimaging Initiative (ADNI). [Online]. Available: http://adni.loni.usc.edu

[5] M. Basereh, M. A. Abikenari, S. Sadeghzadeh, T. Dunn, R. Freichel, P. Siddarth, D. Ghahremani, H. Lavretsky, and V. P. Buch, "ConvNeXt-driven detection of Alzheimer's disease: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes," *Diagnostics*, vol. 15, no. 23, Art. no. 2997, 2025, doi: 10.3390/diagnostics15232997. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC12691013/

[6] M. R. Vyas, "NeuroScan (adni_code)," GitHub repository. [Online]. Available: https://github.com/MeetRVyas/adni_code

[7] W. Jin, Y. Yin, J. Bai, and H. Zhen, "CA-ConvNeXt: Coordinate attention on ConvNeXt for early Alzheimer's disease classification," in *Intelligence Science IV*, IFIP Advances in Information and Communication Technology. Cham, Switzerland: Springer, 2022, pp. 450–457, doi: 10.1007/978-3-031-14903-0_48. [Online]. Available: https://link.springer.com/chapter/10.1007/978-3-031-14903-0_48

[8] M. Basereh *et al.*, "ConvNeXt-driven detection of Alzheimer's disease: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes," bioRxiv preprint, Jul. 2025, doi: 10.1101/2025.07.10.664260. [Online]. Available: https://www.biorxiv.org/content/10.1101/2025.07.10.664260v1.full

[9] M. Basereh, "AlzaSet: MRI dataset for 'ConvNeXt-driven Alzheimer's detection: A benchmark study on expert-annotated AlzaSet MRI dataset across anatomical planes'," Zenodo, version v2, 2025, doi: 10.5281/zenodo.17084199. [Online]. Available: https://zenodo.org/records/17084199

[10] R. Müller, S. Kornblith, and G. E. Hinton, "When does label smoothing help?," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://arxiv.org/abs/1906.02629

[11] G. Xia, O. Laurent, G. Franchi, and C.-S. Bouganis, "Towards understanding why label smoothing degrades selective classification and how to fix it," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025. [Online]. Available: https://arxiv.org/abs/2403.14715

[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Machine Learning (ICML)*, 2017, pp. 1321–1330. [Online]. Available: https://arxiv.org/abs/1706.04599

[13] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626. [Online]. Available: https://arxiv.org/abs/1610.02391

[14] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," arXiv:1802.03426, 2018. [Online]. Available: https://arxiv.org/abs/1802.03426

[15] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456. [Online]. Available: https://arxiv.org/abs/1502.03167

[16] S. Santurkar, D. Tsipras, A. Ilyas, and A. Madry, "How does batch normalization help optimization?," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018. [Online]. Available: https://arxiv.org/abs/1805.11604

[17] M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, "Transfusion: Understanding transfer learning for medical imaging," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://arxiv.org/abs/1902.07208

[18] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019. [Online]. Available: https://arxiv.org/abs/1711.05101

[19] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, "Deep networks with stochastic depth," in *Proc. European Conf. Computer Vision (ECCV)*, 2016, pp. 646–661. [Online]. Available: https://arxiv.org/abs/1603.09382

[20] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random erasing data augmentation," in *Proc. AAAI Conf. Artificial Intelligence*, 2020, pp. 13001–13008. [Online]. Available: https://arxiv.org/abs/1708.04896

[21] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models

[22] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. [Online]. Available: https://arxiv.org/abs/1612.01474
