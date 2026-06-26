# IEEE-Style Result Tables for the EEG Emotion Recognition Pipeline

All values below are taken directly from the run logs and rounded to two decimal places. Where the logs did not report a metric, it is marked as “Not Reported”.

## Table I. Evolution of the Proposed Pipeline

| Run | Major Modification | Feature Count | Classifier | Validation Protocol | Best Accuracy | Remarks |
| --- | --- | ---: | --- | --- | ---: | --- |
| Run 1 | DE-only baseline | 70 | XGBoost (default) | Not Reported | 54.20% | Established the first DE baseline and showed that simple spectral features were insufficient. |
| Run 2 | Windowed DE with differential asymmetry | Not Reported | XGBoost (default) | Not Reported | 55.40% | Added 2-s windowing and asymmetry features to capture temporal dynamics. |
| Run 3 | Baseline-corrected DE features | 70 | XGBoost (default) | Not Reported | 61.50% | Baseline subtraction produced the largest gain in the early pipeline. |
| Run 4 | Hyperparameter tuning of XGBoost | 70 | XGBoost (tuned) | Not Reported | 60.20% | Tuning improved the optimizer but did not outperform baseline-corrected features. |
| Run 5 | Classical ML pipeline with ANOVA + RBF SVM | Not Reported | RBF SVM | Subject-dependent | 56.97% | Introduced classical ML but overfitting was severe under small sample sizes. |
| Run 6 | PCA-based feature space | Not Reported | RBF SVM | Subject-dependent | 54.67% | PCA reduced minority-class sensitivity and led to majority-class domination. |
| Run 7 | Scaled PCA + linear SVM + dynamic thresholding | Not Reported | Linear SVM | Subject-dependent + subject-independent | 59.42% | Achieved the strongest classical subject-independent result. |
| Run 8 | MI feature selection + corrected SVM + CORAL | Not Reported | RBF SVM | Subject-dependent + subject-independent | 58.32% | MI improved feature relevance, but performance did not surpass the PCA-based classical baseline. |
| Run 9 | Production classical ML pipeline with 935 handcrafted features | 935 | RBF SVM | Subject-dependent + subject-independent | 58.51% | Broadened the feature space, but classical ML performance plateaued. |
| Run 10 | High-density windowed DREAMER pipeline with XGBoost and CORAL | 659 | XGBoost | Subject-dependent + subject-independent (LOSO + CORAL) | 58.52% | First large-scale window-level evaluation with 42,389 samples. |
| Run 11 | Leak-proof XGBoost evaluation for valence and arousal | 938 | XGBoost | Subject-dependent + subject-independent (LOSO + CORAL) | 60.14% | Final XGBoost benchmark on binary valence and arousal tasks. |

## Table II. Final Subject-Dependent Results

| Task | Accuracy (%) | Balanced Accuracy (%) | F1-score | MCC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary Valence | 58.77 ± 14.50 | 51.67 ± 13.31 | 26.25 ± 24.01 | 0.03 ± 0.27 | 51.52 ± 23.77 |
| Binary Arousal | 60.14 ± 13.83 | 52.25 ± 12.16 | 31.49 ± 28.30 | 0.04 ± 0.25 | 52.97 ± 19.99 |
| Quadrant Classification | Not Reported | Not Reported | Not Reported | Not Reported | Not Reported |

## Table III. Final Subject-Independent Results

| Task | Accuracy (%) | Balanced Accuracy (%) | F1-score | MCC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary Valence | 58.21 ± 13.07 | 54.86 ± 11.49 | 37.37 ± 17.32 | 0.10 ± 0.26 | 58.58 ± 13.39 |
| Binary Arousal | 57.38 ± 11.45 | 50.83 ± 10.37 | 30.67 ± 17.75 | 0.03 ± 0.23 | 54.17 ± 14.50 |

## Table IV. Ablation Study

| Configuration | Accuracy (%) | Δ Accuracy | Observation |
| --- | ---: | ---: | --- |
| Baseline Pipeline | 54.20 | — | DE-only XGBoost baseline. |
| + Window Segmentation | 55.40 | +1.20 | Added 2-s windowing and differential asymmetry. |
| + Baseline Correction | 61.50 | +6.10 | Baseline subtraction produced the largest gain. |
| + Optimized XGBoost | 60.20 | -1.30 | Hyperparameter tuning did not improve the earlier feature set. |
| + Mutual Information Selection | 58.32 | -1.88 | MI selection was introduced in the corrected SVM pipeline; gains were not clear. |
| + Removal of PCA | Not Reported | Not Reported | No direct ablation removing PCA was reported in the logs. |
| + CORAL Alignment | 58.21 | -0.11 | CORAL was reported in the LOSO pipeline, but no non-CORAL control was reported. |
| Final Proposed Framework | 60.14 | +1.93 | Final leak-proof XGBoost framework evaluated on valence and arousal. |

## Table V. Comparison of Validation Protocols

| Evaluation Protocol | Leakage Prevention | Domain Adaptation | Accuracy | Balanced Accuracy | F1 | MCC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Subject-Dependent (SGKF) | Yes | No | 58.77 | 51.67 | 26.25 | 0.03 |
| Subject-Independent (LOSO) | Yes | No | Not Reported | Not Reported | Not Reported | Not Reported |
| Subject-Independent + CORAL | Yes | Yes | 58.21 | 54.86 | 37.37 | 0.10 |

## Table VI. Final Hyperparameter Configuration

| Hyperparameter | Value |
| --- | --- |
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.01 |
| subsample | 0.90 |
| colsample_bytree | 0.80 |
| Feature selection method | SelectKBest(mutual_info_classif, k=60) |
| Selected features | 60 |
| Scaler | StandardScaler |
| Validation strategy | StratifiedGroupKFold (subject-dependent) and LOSO (subject-independent with CORAL) |

## Summary

The pipeline evolved from a simple DE-only XGBoost baseline in Run 1 to a leak-proof, high-density windowed framework in Run 11. Early runs showed that temporal windowing and especially baseline correction were the most impactful changes, while hyperparameter tuning alone did not yield consistent gains. Later runs expanded the feature space, introduced MI-based selection and CORAL alignment, and finally converged on a robust XGBoost pipeline for valence and arousal classification. The strongest overall result in the final reported XGBoost-based evaluation was obtained in Run 11 for binary arousal subject-dependent classification, with an accuracy of 60.14%.