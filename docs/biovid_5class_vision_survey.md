# Vision-Only Methods for 5-Class Pain Classification on BioVid: A Detailed Survey

> **Last updated:** 2026-03-19
>
> This document surveys all published **vision-only** (facial video/image) methods evaluated on the BioVid Heat Pain Database for **5-class pain intensity classification** (BL/T0, PA1/T1, PA2/T2, PA3/T3, PA4/T4). Chance-level baseline = **20%**.

---

## 1. Overview Comparison Table

| # | Method | Year | Authors | Architecture Type | 5-Class Acc | Binary Acc (BL vs PA4) | Validation |
|---|--------|------|---------|-------------------|-------------|------------------------|------------|
| 1 | **Gkikas 2024 (TNT + Temporal)** | 2024 | Gkikas et al. | Full Transformer | **35.39%** | 77.10% | LOSO |
| 2 | **Transformer (TNT)** | 2023 | Gkikas & Tsiknakis | Transformer-in-Transformer | 31.52% | 73.28% | LOSO |
| 3 | **Face Activity Descriptor** | 2016 | Werner et al. | Handcrafted Features + RF | 30.80% | 72.40% | LOSO |
| 4 | **Facial 3D Distances** | 2016 | Werner et al. | 3D Landmark Distances + RF | 30.30% | 72.10% | LOSO |
| 5 | **ViViT** | 2023 | Benavent-Lledo et al. | Video Vision Transformer | 30.07% | — | Mixed split |
| 6 | **SLSTM** | 2019 | Zhi & Wan | Sparse LSTM | 29.70% | 61.70% | LOSO |
| 7 | **VideoMAE** | 2023 | Benavent-Lledo et al. | Masked Autoencoder | 25.06% | — | Mixed split |
| 8 | **TimeSformer** | 2023 | Benavent-Lledo et al. | Divided Attention | 23.05% | — | Mixed split |

---

## 2. Per-Paper Detailed Analysis

### 2.1 Gkikas et al. 2024 — Full Transformer Framework (Current Vision-Only SOTA)

| Item | Details |
|------|---------|
| **Title** | *Multimodal automatic assessment of acute pain through facial videos and heart rate signals utilizing transformer-based architectures* |
| **Venue** | Frontiers in Pain Research, 5:1372814, 2024 |
| **Authors** | Stefanos Gkikas, Nikolaos S. Tachos, Stelios Andreadis, Vasileios C. Pezoulas, Dimitrios Zaridis, George Gkois, Anastasia Matonaki, Thanos G. Stavropoulos, Dimitrios I. Fotiadis |

#### Architecture

- **Spatial module — Transformer-in-Transformer (TNT):**
  - Input image → patches → sub-patches
  - Inner encoder (4 attention heads) processes sub-patches
  - Outer encoder (10 attention heads) processes patches
  - 12 parallel blocks producing 100-dimensional embeddings
  - Learnable 1D positional encoding
- **Temporal module — Temporal Transformer:**
  - 1 cross-attention block + 3 self-attention blocks (8 heads each)
  - Internal embedding dimension: 128
  - Fourier feature positional encoding
- **Total parameters:** 9.62M (8.60M at inference)

#### Training Details

| Item | Value |
|------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine decay |
| Weight decay | 0.1 |
| Warmup | 50 epochs |
| Batch size | 32 |
| Epochs | 500–800 |
| Pre-training Stage 1 | VGGFace2 (3.31M samples, face recognition) |
| Pre-training Stage 2 | AffectNet + Compound FEE-DB + RAF-DB (emotion recognition, multi-task) |

#### Preprocessing

- Face detection: MTCNN
- Resolution: 448 × 448
- **No facial alignment** (preserves head movement information)

#### Experimental Setup

- **Dataset:** BioVid Heat Pain Database Part A, 87 subjects
- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation
- **Tasks:** 5-class (BL, PA1, PA2, PA3, PA4) and binary (BL vs PA4)

#### Results (Video-Only)

| Task | Accuracy |
|------|----------|
| **5-class** | **35.39%** |
| Binary (BL vs PA4) | 77.10% |

- Also reported multimodal (Video + Heart Rate): 5-class = 39.77%, Binary = 82.74%

#### Key Conclusions

1. Two-stage pre-training (face recognition → emotion recognition) significantly boosts performance
2. Preserving head movement information (no rigid alignment) is better than aligned faces
3. 800 epochs with full augmentation yields best results
4. Temporal transformer provides meaningful improvement over spatial-only

#### Limitations

1. 5-class accuracy is only ~15% above chance despite heavy pre-training and long training
2. Relies on large-scale external face datasets for pre-training (VGGFace2: 3.31M samples)
3. High computational cost (800 epochs training)
4. Multimodal version only marginally better (+4.38% on 5-class), suggesting a ceiling for this approach

---

### 2.2 Gkikas & Tsiknakis 2023 — TNT Spatial Transformer

| Item | Details |
|------|---------|
| **Title** | *A Full Transformer-based Framework for Automatic Pain Estimation using Videos* |
| **Venue** | 45th Annual International Conference of the IEEE EMBC, 2023 |
| **Authors** | Stefanos Gkikas, Manolis Tsiknakis |

#### Architecture

- Same TNT spatial module as the 2024 version (see Section 2.1)
- Simpler temporal module compared to the 2024 extension

#### Training Details

- Same two-stage pre-training strategy: VGGFace2 → emotion recognition datasets
- Specific hyperparameters largely follow the 2024 version

#### Experimental Setup

- BioVid Part A, 87 subjects, LOSO

#### Results

| Task | Accuracy |
|------|----------|
| **5-class** | **31.52%** |
| Binary (BL vs PA4) | 73.28% |

#### Key Conclusions

- First pure-Transformer approach to achieve vision-only SOTA on BioVid at the time
- Two-stage pre-training is critical for performance

#### Limitations

- Short conference paper (4 pages), limited experimental detail
- 5-class accuracy still low

#### Improvement from 2023 → 2024

The 2024 extension improved from 31.52% → 35.39% (+3.87%) primarily through longer training (800 epochs) and improved data augmentation strategies.

---

### 2.3 Werner et al. 2016 — Face Activity Descriptor (FAD)

| Item | Details |
|------|---------|
| **Title** | *Automatic Pain Assessment with Facial Activity Descriptors* |
| **Venue** | IEEE Transactions on Affective Computing, Vol. 8, No. 3, pp. 286–299 |
| **Authors** | Philipp Werner, Ayoub Al-Hamadi, Kerstin Limbrecht-Ecklundt, Steffen Walter, Sascha Gruss, Harald C. Traue |

#### Architecture

- **Not deep learning** — handcrafted feature engineering approach
- **Feature set (FAD):**
  1. Facial landmark distances (2D)
  2. 3D landmark distances
  3. Wrinkle features (average gradient magnitude in nasolabial fold, nose wrinkle regions)
  4. Head pose features
- **Temporal modeling:** Frame-level features → video-level statistical descriptors (mean, max, min, std)
- **Dimensionality reduction:** PCA (retaining 90/95/99% variance)

#### Classifier

- Random Forest (100 trees) — outperformed SVM (RBF kernel)
- Multi-class: one-vs-one strategy

#### Preprocessing

- Subject-specific standardization (normalizing features per subject)

#### Experimental Setup

- BioVid Heat Pain Database + UNBC-McMaster
- LOSO cross-validation

#### Results

| Task | Accuracy |
|------|----------|
| **5-class** | **30.80%** |
| Binary (BL vs PA4) | 72.40% |

#### Ablation — 3D Distances Subset Only

| Task | Accuracy |
|------|----------|
| **5-class** | **30.30%** |
| Binary (BL vs PA4) | 72.10% |

The full FAD set outperforms the 3D distances subset alone, indicating wrinkle and head pose features provide complementary discriminative value.

#### Key Conclusions

1. FAD outperformed all prior methods at the time of publication
2. Temporal integration (frame → video-level statistics) is superior to frame-level decision fusion
3. **Max-value features** (corresponding to facial action apex) and **time-of-max** are the most discriminative
4. Subject-specific normalization is important

#### Limitations

1. Handcrafted features depend on facial landmark detection quality
2. Feature design requires domain expertise
3. Cannot capture deep/abstract features that neural networks learn
4. Despite domain-aware design, 5-class accuracy is only ~11% above chance

---

### 2.4 Benavent-Lledo et al. 2023 — Comprehensive Video Transformer Comparison

| Item | Details |
|------|---------|
| **Title** | *A Comprehensive Study on Pain Assessment from Multimodal Sensor Data* |
| **Venue** | Sensors, 23(24):9675, 2023 |
| **Authors** | Manuel Benavent-Lledo, David Mulero-Perez, David Ortiz-Perez, Javier Rodriguez-Juan, Adrian Berenguer-Agullo, Alexandra Psarrou, Jose Garcia-Rodriguez |

This paper systematically compared **3 video Transformers** and **8 image models**.

#### Models Tested

**Video models:** ViViT, VideoMAE, TimeSformer

**Image models (frame-level):** VGG16 (138M params), MobileNetV2, ResNet50V2, InceptionV3 (23.9M), Xception (22.9M), ViT, BEiT, Swin V2 (3B)

#### Experimental Setup

- BioVid Parts A + B combined (3,480 samples)
- UNBC-McMaster dataset
- **Mixed participant split** (not strict LOSO)

#### Video Model Results — BioVid 5-Class

| Model | Architecture | Accuracy | Test Loss | Train Loss |
|-------|-------------|----------|-----------|------------|
| **ViViT** (Factorized Encoder) | Factorized spatial-then-temporal attention | **30.07%** | 1.49 | — |
| VideoMAE | Masked autoencoder + ViT | 25.06% | 1.58 | 1.56 |
| TimeSformer | Divided space-time attention | 23.05% | 1.60 | 1.64 |

#### Image Model Results — UNBC 4-Class (Mixed Participant Split)

| Model | Accuracy |
|-------|----------|
| BEiT | **96.82%** |
| ViT | 96.37% |
| VGG16 | 94.75% |
| Swin V2 | 93.77% |

> **Critical caveat:** Under mixed participant split, validation = 100% but test = 20% — severe overfitting / data leakage. Frame-level classification on BioVid is unreliable under this split.

#### Key Conclusions

1. **ViViT > VideoMAE > TimeSformer** — Factorized Encoder decomposition strategy is most effective
2. Temporal information did not bring the expected improvement — pain expressions appear only in specific frames
3. All CNNs failed to exceed 20% on 5-class (chance level)
4. Transformer architectures consistently outperform CNNs for frame-level analysis

#### Limitations

1. Non-strict LOSO may overestimate performance
2. BioVid videos contain only facial region — no behavioral or environmental context
3. High inter-class distribution overlap
4. Training details (LR, optimizer, epochs) not fully specified

---

### 2.5 Zhi & Wan 2019 — Sparse LSTM (SLSTM)

| Item | Details |
|------|---------|
| **Title** | *Dynamic Facial Expression Feature Learning Based on Sparse RNN* |
| **Venue** | IEEE 8th Joint International Information Technology and AI Conference (ITAIC), Chongqing, China, pp. 1373–1377 |
| **Authors** | Ruicong Zhi, Ming Wan |

#### Architecture

- **ISTA (Iterative Shrinkage-Thresholding Algorithm)** sparse coding + LSTM
- Core idea: reduce standard LSTM computational complexity via sparse representation while maintaining recognition performance

#### Input

- Facial expression sequence features extracted from video

#### Experimental Setup

- BioVid + CASME II + SMIC datasets
- LOSO cross-validation

#### Results

| Task | Accuracy |
|------|----------|
| **5-class** | **29.70%** |
| Binary (BL vs PA4) | 61.70% |

#### Key Conclusions

- Sparse RNN is computationally more efficient than standard LSTM
- Performance is comparable to standard LSTM with lower complexity

#### Limitations

1. Binary accuracy of only 61.70% is significantly lower than contemporaneous methods (70%+)
2. Sparsity constraints may discard subtle facial dynamics crucial for pain discrimination
3. 5-class accuracy is near the bottom of the leaderboard

---

### 2.6 ErAS-Net (Morsali & Ghaffari 2025) — Multi-Attention + Multi-Patch

| Item | Details |
|------|---------|
| **Title** | *Enhanced residual attention-based subject-specific network (ErAS-Net): facial expression-based pain classification with multiple attention mechanisms* |
| **Venue** | Scientific Reports, 15(1):19425, 2025 |
| **Authors** | Mahdi Morsali, Aboozar Ghaffari |
| **Institution** | Iran University of Science and Technology, Tehran |

> **Note:** This paper did **not** report 5-class results on BioVid, but is included for its cross-dataset evaluation and facial region analysis insights.

#### Architecture

- **Backbone:** Modified ResNet-18 (pre-trained on MsCeleb1M)
- **Input strategy:** 5 parallel streams (1 full face + 4 overlapping patches: top, bottom, left, right)
- **Three-level attention:**
  1. **Input-level spatial attention:** Image divided into 4 overlapping patches
  2. **Feature map spatial attention:** 128 feature maps (28×28) split into left/right halves (28×14), producing two 512-dim feature maps merged in spatial domain
  3. **CBAM modules (Channel & Spatial Attention):** Integrated in ResNet-18 Layer 3 and Layer 4; positioned after second BatchNorm, before residual addition
     - Channel Attention: average + max pooling → MLP bottleneck (reduction ratio r)
     - Spatial Attention: 7×7 convolution on channel-pooled concatenated features
- **Fusion:** Fully Connected Combiner (FC + BN + Sigmoid/SoftMax) — outperformed Weight Optimizer Combiner

#### Training Details

| Item | Value |
|------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early stopping | Patience = 20 epochs |
| Framework | PyTorch |
| Face detection | Dlib (frontal face detector + shape predictor) |
| Preprocessing | Convex hull mask → affine transform alignment → 224×224 |
| Class balancing | Random undersampling of UNBC "no pain" frames |

#### Experimental Setup

| Setting | Details |
|---------|---------|
| UNBC-McMaster | 25 subjects, 200 video sequences, LOSO + 10-fold CV |
| BioVid | 90 participants, **cross-dataset** only (trained on UNBC → tested on BioVid, no fine-tuning) |

#### Results

**UNBC (LOSO, Binary):**

| Metric | Weight Optimizer | FC Combiner |
|--------|-----------------|-------------|
| Accuracy | 86.04% | **89.83%** |
| Precision | 90.82% | 93.05% |
| Recall | 78.74% | 82.98% |
| F1-Score | 81.39% | 86.74% |

Standard deviation reduced from 11.15% (full image only) to 7.34% (FC combiner + patches).

**UNBC (10-fold CV):**

| Task | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| Binary | **98.77%** | 98.35% | 99.07% | 98.54% |
| 4-class | **94.21%** | 93.17% | 93.74% | 93.44% |

**BioVid Cross-Dataset (UNBC-trained → BioVid-tested, no fine-tuning, Binary):**

| Facial Region | Accuracy |
|---------------|----------|
| Top half | **74.48%** |
| Full image | 64.63% |
| Right half | 64.25% |
| Left half | 58.05% |
| Bottom half | 56.23% |
| **Fused (all regions)** | **78.14%** |

#### Key Conclusions

1. **Upper face region** (eyebrows, forehead) carries the most discriminative pain information
2. FC combiner fusion outperforms weight-optimizer fusion
3. Multi-attention reduces subject dependency (lower std across LOSO folds)
4. Cross-dataset generalization (78.14%) demonstrates robustness without fine-tuning

#### Limitations

1. **No 5-class BioVid results reported**
2. Cross-dataset performance (78.14%) is far below within-dataset performance (98.77%)
3. Frame-level only — no temporal modeling
4. Lower face has limited discriminative power (56.23% vs 74.48% for upper face)
5. Dataset accessibility issues noted by authors

---

### 2.7 Bargshady et al. 2024 — ViViT for Binary Pain Detection

| Item | Details |
|------|---------|
| **Title** | *Acute Pain Recognition from Facial Expression Videos using Vision Transformers* |
| **Venue** | 46th Annual International Conference of the IEEE EMBC, 2024 |
| **Authors** | Ghazal Bargshady, Calvin Joseph, Niraj Hirachan, Roland Goecke, Raul Fernandez Rojas |

> **Note:** This paper reported **binary classification only** — no 5-class results.

#### Architecture

- Enhanced Video Vision Transformer (ViViT) for spatio-temporal facial information capture

#### Baselines

- ResNet50
- Hybrid ResNet50 + 3DCNN

#### Experimental Setup

- AI4PAIN Challenge (51 subjects) + BioVid Pain (87 subjects)

#### Results (Binary Only)

| Dataset | Accuracy |
|---------|----------|
| **BioVid** | **79.95%** |
| AI4PAIN | 66.96% |

#### Key Conclusions

- ViViT outperforms ResNet50 and ResNet50+3DCNN on both datasets
- Automated facial expression pain detection has clinical value for patients with communication barriers

#### Limitations

- Short paper (4 pages, EMBC format)
- Binary classification only — no multi-level pain grading
- Specific training hyperparameters not detailed in available abstract

---

## 3. Methodological Dimension Comparison

| Dimension | Handcrafted (Werner 2016) | RNN (SLSTM 2019) | Video Transformer (ViViT et al. 2023) | Full Transformer (Gkikas 2024) | Multi-Attention CNN (ErAS-Net 2025) |
|-----------|--------------------------|-------------------|----------------------------------------|-------------------------------|-------------------------------------|
| **Feature type** | Handcrafted | Learned | Learned | Learned | Learned |
| **Temporal modeling** | Statistical descriptors | LSTM | 3D patch / divided attention | Temporal Transformer | None (frame-level) |
| **Pre-training** | None | None | ImageNet / Kinetics | VGGFace2 + emotion datasets | MsCeleb1M |
| **Parameters** | Very few | Moderate | Millions–tens of millions | 9.62M | ~11M |
| **5-class Acc** | 30.8% | 29.7% | 23–30% | **35.4%** | Not reported |
| **Binary Acc** | 72.4% | 61.7% | 80.0% (Bargshady) | 77.1% | 78.1% (cross-dataset) |
| **Interpretability** | High (semantic features) | Low | Low | Low | Medium (region ablation) |
| **Computational cost** | Low | Medium | High | High | Medium |

---

## 4. Common Limitations Across All Methods

| Limitation | Description |
|------------|-------------|
| **Label noise** | BioVid labels are based on standardized thermal stimulation protocol (T0–T4), not subjective pain ratings. Individual pain thresholds vary, causing mismatch between labels and actual experience. |
| **Inter-class overlap** | Adjacent pain levels (PA1 vs PA2 vs PA3) produce nearly indistinguishable facial expressions. Intra-subject variation across sessions often exceeds inter-class variation. |
| **Information bottleneck** | Videos contain only frontal face, with 5.5-second windows. Pain-indicative expressions may appear in only a few frames. |
| **Generalization gap** | 87 healthy young subjects with acute experimentally-induced pain ≠ clinical chronic pain populations. |
| **LOSO stringency** | LOSO is the gold standard but significantly lowers absolute numbers. Non-LOSO results (e.g., Benavent-Lledo's 97%) exhibit severe overfitting and should not be compared directly. |
| **Modality ceiling** | Vision-only methods plateau at ~35%. Multimodal fusion (+ECG/EMG/GSR) reaches ~38–40%, suggesting facial visual information alone is insufficient. |
| **Dataset homogeneity** | All subjects are healthy adults from a narrow demographic. No clinical pain, no diverse populations, no chronic conditions. |
| **Ecological validity** | Controlled lab setting (frontal view, good lighting, fixed camera) does not represent real clinical environments. |

---

## 5. Key Takeaways

1. **5-class pain classification on BioVid is an extremely hard task.** The current vision-only SOTA (Gkikas 2024) is only **35.39%** under LOSO — merely 15% above chance.

2. **Two-stage pre-training is the most impactful technique.** Gkikas' face recognition → emotion recognition pre-training pipeline accounts for most of the performance gain over methods without domain-specific pre-training.

3. **Handcrafted features remain competitive.** Werner et al. (2016) achieved 30.8% with FAD — only 4.6% below the 2024 deep learning SOTA — suggesting that the bottleneck is in the data/task, not in feature extraction capacity.

4. **Temporal modeling helps but not dramatically.** ViViT (30.07%) slightly outperforms frame-level methods, and the temporal Transformer in Gkikas (35.39%) adds ~4% over spatial-only (31.52%). Pain expressions are too brief and subtle for temporal context to provide large gains.

5. **Upper face carries the most pain information.** ErAS-Net's ablation shows upper face (eyebrows, forehead) alone achieves 74.48% binary accuracy vs. 56.23% for lower face — consistent with AU4 (brow lowerer) and AU43 (eye closure) being key pain indicators.

6. **The fundamental barrier is biological, not algorithmic.** Individual variation in pain expression and pain threshold exceeds the signal from pain intensity levels, making 5-class discrimination inherently difficult from vision alone.

---

## 6. References

1. Gkikas, S., Tachos, N. S., Andreadis, S., et al. (2024). Multimodal automatic assessment of acute pain through facial videos and heart rate signals utilizing transformer-based architectures. *Frontiers in Pain Research*, 5:1372814. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004333/)

2. Gkikas, S., & Tsiknakis, M. (2023). A Full Transformer-based Framework for Automatic Pain Estimation using Videos. *45th IEEE EMBC*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38083481/)

3. Werner, P., Al-Hamadi, A., Limbrecht-Ecklundt, K., Walter, S., Gruss, S., & Traue, H. C. (2016). Automatic Pain Assessment with Facial Activity Descriptors. *IEEE Trans. Affective Computing*, 8(3), 286–299. [Link](https://dl.acm.org/doi/abs/10.1109/TAFFC.2016.2537327)

4. Benavent-Lledo, M., Mulero-Perez, D., Ortiz-Perez, D., et al. (2023). A Comprehensive Study on Pain Assessment from Multimodal Sensor Data. *Sensors*, 23(24):9675. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10747670/)

5. Zhi, R., & Wan, M. (2019). Dynamic Facial Expression Feature Learning Based on Sparse RNN. *IEEE ITAIC*, pp. 1373–1377. [Link](https://ieeexplore.ieee.org/document/8785844/)

6. Morsali, M., & Ghaffari, A. (2025). Enhanced residual attention-based subject-specific network (ErAS-Net). *Scientific Reports*, 15(1):19425. [Link](https://www.nature.com/articles/s41598-025-04552-w)

7. Bargshady, G., Joseph, C., Hirachan, N., Goecke, R., & Fernandez Rojas, R. (2024). Acute Pain Recognition from Facial Expression Videos using Vision Transformers. *46th IEEE EMBC*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40039359/)

8. Nguyen, M.-D., Yang, H.-J., Dao, D.-P., et al. (2025). Dual-stream transformer approach for pain assessment using visual-physiological data modeling. *PeerJ Computer Science*. [Link](https://peerj.com/articles/cs-3158/)
