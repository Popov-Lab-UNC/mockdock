# Top Showcase Docking Poses (Geometric Mean Selection)

This document lists the best-performing molecular docking poses generated across all benchmark targets and generative models. The candidates are selected using a **Cumulative Score** calculated as the **Geometric Mean** of six normalized metrics:
1. **Docking Score** (AutoDock / Gnina binding energy; higher is better when normalized. Already normalized in results as `norm_score`)
2. **QED** (Quantitative Estimate of Drug-likeness; range $[0, 1]$, higher is better)
3. **SA** (Synthetic Accessibility; range $[1, 10]$, lower is better, normalized using $\frac{10 - SA}{9}$)
4. **MolSkill Score** (Chemist organic synthetic preference; range $[-30, 40]$, lower is better, normalized using $\frac{40 - MolSkill}{70}$)
5. **STOPLIGHT Score** (ADMET toxicity risk; range $[0, 2]$, lower is better, normalized using $\frac{2 - STOPLIGHT}{2}$)
6. **AIZynthFinder State Score** (Route feasibility; range $[0, 1]$, higher is better)

### Normalization Bounds & Scaling
| Metric | Original Range / Condition | Best Value | Normalization Mapping |
| :--- | :--- | :--- | :--- |
| **Docking Score** | Already scaled as `norm_score` | High | Direct value (Clipped to $\ge 0$, uncapped above $1.0$) |
| **QED** | $[0, 1]$ | $1.0$ | Direct value (Clipped to $[0, 1]$) |
| **SA** | $[1, 10]$ | $1.0$ | $\frac{10 - SA}{9}$ (Clipped to $[0, 1]$) |
| **MolSkill** | Unbounded (Scaled $[-30, 40]$) | $-30.0$ | $\frac{40 - MolSkill}{70}$ (Clipped to $[0, 1]$) |
| **STOPLIGHT**| $[0, 2]$ | $0.0$ | $\frac{2 - STOPLIGHT}{2}$ (Clipped to $[0, 1]$) |
| **AIZynthFinder**| $[0, 1]$ | $1.0$ | Direct state score (Clipped to $[0, 1]$) |

*The geometric mean is calculated over all available normalized metrics. For stability, normalized values are clamped to a minimum of $10^{-4}$ prior to calculating the geometric mean. Note that in Uncapped runs, the Docking Score component can exceed $1.0$ (signifying optimization exceeding the initial benchmark target threshold), allowing both the Normalized Docking and the Geometric Mean Cumulative Score to go above $1.0$.*

---

## Model: A2C

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.745** | -9.92 | 0.650 | 0.745 | 2.20 | 7.39 | 0.17 | 0.956 | `exps_upperbound/acegen-a2c/run_20260429_210431_r02/CHK1/poses/chunk_128_lig_2_s0.dlg` | 8 |
| Capped | DPP4 | **0.841** | -11.33 | 1.014 | 0.855 | 3.03 | -9.69 | 0.08 | 0.773 | `exps_upperbound/acegen-a2c/run_20260429_210431_r04/DPP4/poses/chunk_112_lig_1_s0.dlg` | 7 |
| Capped | ITK | **0.908** | -10.61 | 1.213 | 0.849 | 2.38 | -11.86 | 0.25 | 0.994 | `exps_upperbound/acegen-a2c/run_20260429_210431_r03/ITK/poses/chunk_320_lig_7_s0.dlg` | 7 |
| Capped | PEPCK | **0.771** | -11.84 | 1.019 | 0.462 | 2.49 | -6.71 | 0.33 | 0.963 | `exps_upperbound/acegen-a2c/run_20260429_210431_r03/PEPCK/poses/chunk_128_lig_9_s0.dlg` | 0 |
| Capped | PptT | **0.694** | -12.01 | 0.310 | 0.778 | 2.21 | -5.25 | 0.33 | 0.994 | `exps_upperbound/acegen-a2c/run_20260429_210431_r02/PptT/poses/chunk_750_lig_9_s1.dlg` | 2 |
| Capped | TTK | **0.796** | -11.68 | 0.966 | 0.605 | 2.62 | -5.01 | 0.33 | 0.987 | `exps_upperbound/acegen-a2c/run_20260429_210431_r03/TTK/poses/chunk_416_lig_2_s0.dlg` | 5 |
| Uncapped | CHK1 | **0.755** | -10.66 | 0.789 | 0.645 | 2.99 | -3.14 | 0.17 | 0.832 | `exps/acegen-a2c/run_20260430_215420_r05/CHK1/poses/chunk_0_lig_1_s0.dlg` | 5 |
| Uncapped | DPP4 | **0.860** | -11.17 | 0.951 | 0.858 | 2.94 | -12.46 | 0.25 | 0.963 | `exps/acegen-a2c/run_20260430_215420_r04/DPP4/poses/chunk_958_lig_5_s0.dlg` | 2 |
| Uncapped | ITK | **0.899** | -11.22 | 1.465 | 0.712 | 2.65 | -10.43 | 0.25 | 0.987 | `exps/acegen-a2c/run_20260430_215420_r02/ITK/poses/chunk_128_lig_29_s0.dlg` | 6 |
| Uncapped | PEPCK | **0.767** | -11.79 | 0.999 | 0.461 | 2.55 | -1.93 | 0.17 | 0.975 | `exps/acegen-a2c/run_20260430_215420_r05/PEPCK/poses/chunk_370_lig_44_s0.dlg` | 6 |
| Uncapped | PptT | **0.666** | -15.86 | 0.743 | 0.500 | 2.84 | N/A | N/A | N/A | `exps/acegen-a2c/run_20260430_215420_r02/PptT/poses/chunk_0_lig_11_s1.dlg` | 6 |
| Uncapped | TTK | **0.810** | -10.36 | 0.631 | 0.833 | 2.54 | -9.85 | 0.17 | 0.994 | `exps/acegen-a2c/run_20260430_215420_r01/TTK/poses/chunk_384_lig_22_s0.dlg` | 2 |
| Uncapped | VEGFR2 | **0.647** | -10.06 | 0.174 | 0.728 | 2.80 | -17.45 | 0.17 | 0.963 | `exps/acegen-a2c/run_20260430_215420_r03/VEGFR2/poses/chunk_510_lig_0_s0.dlg` | 5 |

## Model: AHC

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.782** | -10.71 | 0.798 | 0.631 | 2.55 | -1.86 | 0.08 | 0.952 | `exps_upperbound/acegen-ahc/run_20260429_210505_r04/CHK1/poses/chunk_750_lig_38_s0.dlg` | 9 |
| Capped | DPP4 | **0.871** | -11.43 | 1.054 | 0.744 | 3.34 | -16.33 | 0.08 | 0.975 | `exps_upperbound/acegen-ahc/run_20260429_210505_r01/DPP4/poses/chunk_638_lig_29_s0.dlg` | 8 |
| Capped | ITK | **0.882** | -12.54 | 2.009 | 0.571 | 2.68 | -3.78 | 0.33 | 0.963 | `exps_upperbound/acegen-ahc/run_20260429_210505_r02/ITK/poses/chunk_640_lig_39_s0.dlg` | 4 |
| Capped | PEPCK | **0.789** | -13.08 | 1.509 | 0.429 | 2.57 | 1.21 | 0.33 | 0.975 | `exps_upperbound/acegen-ahc/run_20260429_210505_r01/PEPCK/poses/chunk_359_lig_36_s1.dlg` | 5 |
| Capped | PptT | **0.693** | -11.42 | 0.244 | 0.731 | 2.31 | -18.35 | 0.25 | 0.994 | `exps_upperbound/acegen-ahc/run_20260429_210505_r02/PptT/poses/chunk_947_lig_2_s2.dlg` | 1 |
| Capped | TTK | **0.800** | -10.42 | 0.647 | 0.816 | 2.40 | -11.03 | 0.33 | 0.963 | `exps_upperbound/acegen-ahc/run_20260429_210505_r01/TTK/poses/chunk_384_lig_33_s0.dlg` | 0 |
| Capped | VEGFR2 | **0.185** | -7.48 | 0.000 | 0.809 | 2.60 | -9.37 | 0.25 | 0.987 | `exps_upperbound/acegen-ahc/run_20260429_210505_r02/VEGFR2/poses/chunk_744_lig_1_s0.dlg` | 8 |
| Uncapped | CHK1 | **0.756** | -10.68 | 0.793 | 0.645 | 2.99 | -3.14 | 0.17 | 0.832 | `exps/acegen-ahc/run_20260430_215906_r05/CHK1/poses/chunk_0_lig_1_s0.dlg` | 1 |
| Uncapped | DPP4 | **0.851** | -11.94 | 1.257 | 0.835 | 3.33 | -9.18 | 0.08 | 0.726 | `exps/acegen-ahc/run_20260430_215906_r03/DPP4/poses/chunk_576_lig_8_s0.dlg` | 7 |
| Uncapped | ITK | **0.880** | -10.70 | 1.250 | 0.764 | 2.70 | -9.85 | 0.25 | 0.963 | `exps/acegen-ahc/run_20260430_215906_r01/ITK/poses/chunk_640_lig_20_s2.dlg` | 3 |
| Uncapped | PEPCK | **0.766** | -11.76 | 0.987 | 0.461 | 2.55 | -1.93 | 0.17 | 0.975 | `exps/acegen-ahc/run_20260430_215906_r02/PEPCK/poses/chunk_58_lig_19_s0.dlg` | 4 |
| Uncapped | PptT | **0.696** | -12.78 | 0.396 | 0.717 | 2.46 | -10.18 | 0.33 | 0.797 | `exps/acegen-ahc/run_20260430_215906_r02/PptT/poses/chunk_948_lig_1_s1.dlg` | 8 |
| Uncapped | TTK | **0.799** | -10.92 | 0.773 | 0.728 | 2.53 | -9.29 | 0.42 | 0.998 | `exps/acegen-ahc/run_20260430_215906_r05/TTK/poses/chunk_319_lig_10_s0.dlg` | 6 |
| Uncapped | VEGFR2 | **0.184** | -4.81 | 0.000 | 0.741 | 3.30 | -11.53 | 0.00 | 0.963 | `exps/acegen-ahc/run_20260430_215906_r03/VEGFR2/poses/chunk_64_lig_0_s0.dlg` | 3 |

## Model: GenMol

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.749** | -9.65 | 0.600 | 0.728 | 2.48 | -6.64 | 0.17 | 0.794 | `exps_upperbound/genmol/run_20260527_011437_r04/CHK1/poses/chunk_320_lig_0_s0.dlg` | 2 |
| Capped | DPP4 | **0.828** | -10.57 | 0.712 | 0.903 | 2.80 | -14.92 | 0.33 | 0.956 | `exps_upperbound/genmol/run_20260527_011437_r03/DPP4/poses/chunk_320_lig_3_s0.dlg` | 6 |
| Capped | ITK | **0.853** | -10.99 | 1.370 | 0.648 | 2.44 | -6.43 | 0.42 | 0.987 | `exps_upperbound/genmol/run_20260527_011437_r01/ITK/poses/chunk_253_lig_36_s0.dlg` | 3 |
| Capped | PEPCK | **0.756** | -12.13 | 1.134 | 0.433 | 2.67 | -3.39 | 0.42 | 0.956 | `exps_upperbound/genmol/run_20260527_011437_r05/PEPCK/poses/chunk_192_lig_0_s1.dlg` | 2 |
| Capped | PptT | **0.753** | -13.34 | 0.459 | 0.690 | 2.60 | -13.67 | 0.17 | 0.998 | `exps_upperbound/genmol/run_20260527_011437_r02/PptT/poses/chunk_896_lig_22_s1.dlg` | 0 |
| Capped | TTK | **0.803** | -10.85 | 0.756 | 0.713 | 2.14 | -5.79 | 0.25 | 0.994 | `exps_upperbound/genmol/run_20260527_011437_r01/TTK/poses/chunk_832_lig_40_s0.dlg` | 7 |
| Capped | VEGFR2 | **0.763** | -12.89 | 1.169 | 0.423 | 2.75 | -7.25 | 0.50 | 0.975 | `exps_upperbound/genmol/run_20260527_011437_r04/VEGFR2/poses/chunk_895_lig_37_s2.dlg` | 2 |
| Uncapped | CHK1 | **0.765** | -10.94 | 0.841 | 0.677 | 2.64 | -5.54 | 0.33 | 0.794 | `exps/genmol/run_20260527_070234_r04/CHK1/poses/chunk_576_lig_15_s1.dlg` | 2 |
| Uncapped | DPP4 | **0.878** | -11.67 | 1.149 | 0.855 | 2.85 | -11.48 | 0.33 | 0.956 | `exps/genmol/run_20260527_070234_r03/DPP4/poses/chunk_960_lig_7_s0.dlg` | 7 |
| Uncapped | ITK | **0.899** | -11.22 | 1.465 | 0.770 | 2.66 | -7.23 | 0.25 | 0.975 | `exps/genmol/run_20260527_070234_r01/ITK/poses/chunk_893_lig_28_s0.dlg` | 5 |
| Uncapped | PEPCK | **0.750** | -11.43 | 0.857 | 0.510 | 2.45 | -3.76 | 0.42 | 0.987 | `exps/genmol/run_20260527_070234_r03/PEPCK/poses/chunk_640_lig_2_s0.dlg` | 2 |
| Uncapped | PptT | **0.741** | -12.67 | 0.384 | 0.751 | 2.16 | -12.83 | 0.25 | 0.994 | `exps/genmol/run_20260527_070234_r02/PptT/poses/chunk_832_lig_21_s1.dlg` | 9 |
| Uncapped | TTK | **0.807** | -11.71 | 0.974 | 0.677 | 2.94 | -1.19 | 0.17 | 0.994 | `exps/genmol/run_20260527_070234_r02/TTK/poses/chunk_64_lig_8_s0.dlg` | 1 |
| Uncapped | VEGFR2 | **0.758** | -13.31 | 1.316 | 0.431 | 2.60 | -0.14 | 0.58 | 0.994 | `exps/genmol/run_20260527_070234_r02/VEGFR2/poses/chunk_960_lig_16_s0.dlg` | 5 |

## Model: InVirtuoGen

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.651** | -9.71 | 0.611 | 0.463 | 2.28 | 5.11 | 0.42 | 0.794 | `exps_upperbound/invirtuogen/run_20260624_095031_r05/CHK1/poses/chunk_931_lig_14_s0.dlg` | 0 |
| Capped | DPP4 | **0.815** | -11.74 | 1.177 | 0.714 | 3.54 | -7.58 | 0.25 | 0.815 | `exps_upperbound/invirtuogen/run_20260624_095031_r02/DPP4/poses/chunk_231_lig_47_s0.dlg` | 7 |
| Capped | ITK | **0.824** | -10.36 | 1.110 | 0.837 | 3.50 | -2.14 | 0.00 | 0.773 | `exps_upperbound/invirtuogen/run_20260624_095031_r02/ITK/poses/chunk_11_lig_4_s0.dlg` | 0 |
| Capped | PEPCK | **0.727** | -12.13 | 1.134 | 0.594 | 4.18 | -1.92 | 0.42 | 0.718 | `exps_upperbound/invirtuogen/run_20260624_095031_r05/PEPCK/poses/chunk_90_lig_0_s0.dlg` | 7 |
| Capped | PptT | **0.660** | -11.61 | 0.265 | 0.751 | 2.31 | 2.56 | 0.17 | 0.994 | `exps_upperbound/invirtuogen/run_20260624_095031_r04/PptT/poses/chunk_11_lig_1_s1.dlg` | 6 |
| Capped | TTK | **0.767** | -10.85 | 0.756 | 0.686 | 3.40 | -11.82 | 0.33 | 0.865 | `exps_upperbound/invirtuogen/run_20260624_095031_r01/TTK/poses/chunk_0_lig_8_s0.dlg` | 7 |
| Capped | VEGFR2 | **0.777** | -11.84 | 0.800 | 0.799 | 3.08 | -0.04 | 0.42 | 0.994 | `exps_upperbound/invirtuogen/run_20260624_095031_r01/VEGFR2/poses/chunk_371_lig_2_s0.dlg` | 4 |
| Uncapped | CHK1 | **0.653** | -9.96 | 0.658 | 0.436 | 2.33 | 4.76 | 0.42 | 0.794 | `exps/invirtuogen/run_20260624_095011_r05/CHK1/poses/chunk_511_lig_11_s0.dlg` | 4 |
| Uncapped | DPP4 | **0.808** | -11.04 | 0.899 | 0.774 | 3.17 | -14.38 | 0.33 | 0.815 | `exps/invirtuogen/run_20260624_095011_r05/DPP4/poses/chunk_511_lig_3_s0.dlg` | 8 |
| Uncapped | ITK | **0.824** | -10.36 | 1.110 | 0.837 | 3.50 | -2.14 | 0.00 | 0.773 | `exps/invirtuogen/run_20260624_095011_r02/ITK/poses/chunk_11_lig_4_s0.dlg` | 6 |
| Uncapped | PEPCK | **1.462** | -12.96 | 1.462 | N/A | N/A | N/A | N/A | N/A | `exps/invirtuogen/run_20260624_095011_r04/PEPCK/poses/chunk_651_lig_9_s6.dlg` | 6 |
| Uncapped | PptT | **0.666** | -11.73 | 0.278 | 0.751 | 2.31 | 2.56 | 0.17 | 0.994 | `exps/invirtuogen/run_20260624_095011_r04/PptT/poses/chunk_11_lig_1_s1.dlg` | 3 |
| Uncapped | TTK | **0.766** | -10.82 | 0.748 | 0.686 | 3.40 | -11.82 | 0.33 | 0.865 | `exps/invirtuogen/run_20260624_095011_r01/TTK/poses/chunk_0_lig_8_s0.dlg` | 9 |
| Uncapped | VEGFR2 | **0.771** | -11.76 | 0.772 | 0.526 | 3.27 | -17.32 | 0.25 | 0.963 | `exps/invirtuogen/run_20260624_095011_r03/VEGFR2/poses/chunk_371_lig_9_s0.dlg` | 0 |

## Model: LibINVENT

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.738** | -13.04 | 1.234 | 0.497 | 2.99 | 6.11 | 0.25 | 0.798 | `exps_upperbound/libinvent/run_20260526_220933_r01/CHK1/poses/chunk_877_lig_33_s0.dlg` | 2 |
| Capped | DPP4 | **0.863** | -13.25 | 1.777 | 0.664 | 3.37 | -8.91 | 0.33 | 0.817 | `exps_upperbound/libinvent/run_20260526_220933_r02/DPP4/poses/chunk_635_lig_39_s0.dlg` | 9 |
| Capped | ITK | **0.912** | -10.89 | 1.328 | 0.790 | 2.41 | -16.00 | 0.33 | 0.975 | `exps_upperbound/libinvent/run_20260526_220933_r05/ITK/poses/chunk_634_lig_12_s0.dlg` | 0 |
| Capped | PEPCK | **0.753** | -12.24 | 1.177 | 0.406 | 2.73 | -2.35 | 0.42 | 0.987 | `exps_upperbound/libinvent/run_20260526_220933_r04/PEPCK/poses/chunk_40_lig_33_s0.dlg` | 5 |
| Capped | PptT | **0.740** | -15.99 | 0.757 | 0.668 | 2.37 | 1.51 | 0.33 | 0.834 | `exps_upperbound/libinvent/run_20260526_220933_r01/PptT/poses/chunk_694_lig_4_s1.dlg` | 9 |
| Capped | TTK | **0.807** | -11.67 | 0.963 | 0.709 | 3.07 | -6.91 | 0.17 | 0.857 | `exps_upperbound/libinvent/run_20260526_220933_r03/TTK/poses/chunk_658_lig_25_s0.dlg` | 2 |
| Capped | VEGFR2 | **0.802** | -12.12 | 0.898 | 0.632 | 2.71 | -7.37 | 0.25 | 0.975 | `exps_upperbound/libinvent/run_20260526_220933_r03/VEGFR2/poses/chunk_291_lig_45_s0.dlg` | 2 |
| Uncapped | CHK1 | **0.735** | -10.22 | 0.707 | 0.692 | 2.67 | -3.81 | 0.25 | 0.726 | `exps/libinvent/run_20260527_023257_r04/CHK1/poses/chunk_0_lig_5_s1.dlg` | 3 |
| Uncapped | DPP4 | **0.847** | -11.39 | 1.038 | 0.934 | 2.91 | -9.75 | 0.33 | 0.817 | `exps/libinvent/run_20260527_023257_r05/DPP4/poses/chunk_962_lig_27_s0.dlg` | 6 |
| Uncapped | ITK | **0.923** | -10.34 | 1.101 | 0.928 | 2.33 | -14.25 | 0.08 | 0.952 | `exps/libinvent/run_20260527_023257_r01/ITK/poses/chunk_450_lig_18_s0.dlg` | 9 |
| Uncapped | PEPCK | **0.739** | -12.66 | 1.343 | 0.377 | 2.75 | -1.47 | 0.58 | 0.952 | `exps/libinvent/run_20260527_023257_r04/PEPCK/poses/chunk_603_lig_38_s0.dlg` | 9 |
| Uncapped | PptT | **0.776** | -15.35 | 0.685 | 0.633 | 2.33 | -12.71 | 0.42 | 0.994 | `exps/libinvent/run_20260527_023257_r01/PptT/poses/chunk_827_lig_13_s6.dlg` | 2 |
| Uncapped | TTK | **0.783** | -10.28 | 0.611 | 0.867 | 2.55 | 1.20 | 0.08 | 0.987 | `exps/libinvent/run_20260527_023257_r03/TTK/poses/chunk_506_lig_32_s0.dlg` | 7 |
| Uncapped | VEGFR2 | **0.824** | -12.71 | 1.105 | 0.629 | 2.55 | -5.84 | 0.33 | 0.994 | `exps/libinvent/run_20260527_023257_r02/VEGFR2/poses/chunk_405_lig_37_s1.dlg` | 6 |

## Model: PPO

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.755** | -10.50 | 0.759 | 0.745 | 2.29 | 9.52 | 0.17 | 0.956 | `exps_upperbound/acegen-ppo/run_20260429_210505_r04/CHK1/poses/chunk_360_lig_4_s0.dlg` | 4 |
| Capped | DPP4 | **0.854** | -11.65 | 1.141 | 0.786 | 3.02 | -14.37 | 0.17 | 0.785 | `exps_upperbound/acegen-ppo/run_20260429_210505_r03/DPP4/poses/chunk_320_lig_34_s0.dlg` | 9 |
| Capped | ITK | **0.934** | -11.08 | 1.407 | 0.814 | 2.32 | -12.36 | 0.17 | 0.994 | `exps_upperbound/acegen-ppo/run_20260429_210505_r01/ITK/poses/chunk_448_lig_24_s0.dlg` | 5 |
| Capped | PEPCK | **0.785** | -13.01 | 1.481 | 0.429 | 2.57 | 1.21 | 0.33 | 0.963 | `exps_upperbound/acegen-ppo/run_20260430_113110_r01/PEPCK/poses/chunk_698_lig_49_s1.dlg` | 1 |
| Capped | PptT | **0.670** | -15.99 | 0.757 | 0.500 | 2.84 | N/A | N/A | N/A | `exps_upperbound/acegen-ppo/run_20260429_210505_r02/PptT/poses/chunk_0_lig_11_s1.dlg` | 9 |
| Capped | TTK | **0.809** | -10.51 | 0.669 | 0.818 | 2.99 | -12.76 | 0.17 | 0.951 | `exps_upperbound/acegen-ppo/run_20260429_210505_r01/TTK/poses/chunk_831_lig_36_s0.dlg` | 6 |
| Capped | VEGFR2 | **0.619** | -10.62 | 0.371 | 0.478 | 2.68 | -3.81 | 0.42 | 0.785 | `exps_upperbound/acegen-ppo/run_20260429_210505_r01/VEGFR2/poses/chunk_509_lig_0_s0.dlg` | 0 |
| Uncapped | CHK1 | **0.760** | -10.83 | 0.821 | 0.643 | 2.44 | 7.17 | 0.08 | 0.963 | `exps/acegen-ppo/run_20260430_215905_r05/CHK1/poses/chunk_614_lig_0_s0.dlg` | 9 |
| Uncapped | DPP4 | **0.872** | -11.76 | 1.185 | 0.873 | 3.03 | -10.42 | 0.08 | 0.793 | `exps/acegen-ppo/run_20260430_215905_r04/DPP4/poses/chunk_703_lig_40_s0.dlg` | 2 |
| Uncapped | ITK | **0.880** | -10.75 | 1.271 | 0.759 | 2.18 | -11.05 | 0.42 | 0.963 | `exps/acegen-ppo/run_20260430_215905_r03/ITK/poses/chunk_64_lig_20_s0.dlg` | 4 |
| Uncapped | PEPCK | **0.787** | -13.03 | 1.489 | 0.429 | 2.57 | 1.21 | 0.33 | 0.975 | `exps/acegen-ppo/run_20260430_215905_r01/PEPCK/poses/chunk_561_lig_27_s1.dlg` | 8 |
| Uncapped | PptT | **0.670** | -15.99 | 0.757 | 0.500 | 2.84 | N/A | N/A | N/A | `exps/acegen-ppo/run_20260430_215905_r03/PptT/poses/chunk_0_lig_11_s1.dlg` | 4 |
| Uncapped | TTK | **0.838** | -12.81 | 1.253 | 0.654 | 3.27 | -9.10 | 0.33 | 0.963 | `exps/acegen-ppo/run_20260430_215905_r01/TTK/poses/chunk_384_lig_4_s1.dlg` | 6 |
| Uncapped | VEGFR2 | **0.595** | -10.40 | 0.294 | 0.534 | 4.57 | -7.00 | 0.17 | 0.766 | `exps/acegen-ppo/run_20260430_215905_r05/VEGFR2/poses/chunk_512_lig_0_s0.dlg` | 2 |

## Model: PPOD

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.761** | -10.24 | 0.710 | 0.695 | 2.35 | 4.92 | 0.08 | 0.963 | `exps_upperbound/acegen-ppod/run_20260429_211332_r04/CHK1/poses/chunk_821_lig_7_s2.dlg` | 0 |
| Capped | DPP4 | **0.872** | -11.76 | 1.185 | 0.873 | 3.03 | -10.42 | 0.08 | 0.793 | `exps_upperbound/acegen-ppod/run_20260429_211332_r04/DPP4/poses/chunk_896_lig_2_s0.dlg` | 0 |
| Capped | ITK | **0.890** | -10.84 | 1.308 | 0.757 | 2.70 | -12.36 | 0.33 | 0.994 | `exps_upperbound/acegen-ppod/run_20260429_211332_r04/ITK/poses/chunk_384_lig_24_s4.dlg` | 8 |
| Capped | PEPCK | **0.787** | -13.07 | 1.505 | 0.429 | 2.57 | 1.21 | 0.33 | 0.963 | `exps_upperbound/acegen-ppod/run_20260429_211332_r05/PEPCK/poses/chunk_876_lig_11_s1.dlg` | 5 |
| Capped | PptT | **0.683** | -17.02 | 0.873 | 0.459 | 2.70 | -0.36 | 0.58 | 0.761 | `exps_upperbound/acegen-ppod/run_20260429_211332_r05/PptT/poses/chunk_842_lig_7_s14.dlg` | 6 |
| Capped | TTK | **0.828** | -10.91 | 0.771 | 0.812 | 2.62 | -10.35 | 0.17 | 0.956 | `exps_upperbound/acegen-ppod/run_20260429_211332_r04/TTK/poses/chunk_768_lig_21_s0.dlg` | 8 |
| Capped | VEGFR2 | **0.585** | -10.16 | 0.209 | 0.528 | 2.55 | -3.40 | 0.25 | 0.805 | `exps_upperbound/acegen-ppod/run_20260429_211332_r03/VEGFR2/poses/chunk_443_lig_0_s0.dlg` | 8 |
| Uncapped | CHK1 | **0.775** | -9.52 | 0.576 | 0.723 | 2.30 | -8.26 | 0.17 | 0.963 | `exps/acegen-ppod/run_20260430_220015_r03/CHK1/poses/chunk_247_lig_24_s0.dlg` | 3 |
| Uncapped | DPP4 | **0.858** | -11.13 | 0.935 | 0.858 | 2.94 | -12.46 | 0.25 | 0.963 | `exps/acegen-ppod/run_20260430_220015_r04/DPP4/poses/chunk_320_lig_32_s0.dlg` | 2 |
| Uncapped | ITK | **2.174** | -12.94 | 2.174 | N/A | N/A | N/A | N/A | N/A | `exps/acegen-ppod/run_20260430_220015_r05/ITK/poses/chunk_574_lig_18_s3.dlg` | 9 |
| Uncapped | PEPCK | **0.767** | -11.78 | 0.995 | 0.461 | 2.55 | -1.93 | 0.17 | 0.975 | `exps/acegen-ppod/run_20260430_220015_r01/PEPCK/poses/chunk_440_lig_14_s0.dlg` | 3 |
| Uncapped | PptT | **0.670** | -15.99 | 0.757 | 0.500 | 2.84 | N/A | N/A | N/A | `exps/acegen-ppod/run_20260430_220015_r05/PptT/poses/chunk_0_lig_11_s1.dlg` | 2 |
| Uncapped | TTK | **0.829** | -10.84 | 0.753 | 0.772 | 2.95 | -12.82 | 0.08 | 0.987 | `exps/acegen-ppod/run_20260430_220015_r01/TTK/poses/chunk_896_lig_39_s0.dlg` | 6 |
| Uncapped | VEGFR2 | **0.179** | -7.65 | 0.000 | 0.665 | 2.33 | -8.85 | 0.33 | 0.998 | `exps/acegen-ppod/run_20260430_220015_r03/VEGFR2/poses/chunk_510_lig_0_s0.dlg` | 5 |

## Model: REINFORCE

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.777** | -10.31 | 0.723 | 0.688 | 2.49 | -0.56 | 0.08 | 0.951 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r05/CHK1/poses/chunk_0_lig_15_s0.dlg` | 3 |
| Capped | DPP4 | **0.849** | -12.69 | 1.555 | 0.693 | 2.89 | -5.02 | 0.33 | 0.820 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r03/DPP4/poses/chunk_895_lig_37_s0.dlg` | 6 |
| Capped | ITK | **0.893** | -11.48 | 1.572 | 0.741 | 2.62 | -6.83 | 0.33 | 0.952 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r04/ITK/poses/chunk_768_lig_9_s0.dlg` | 2 |
| Capped | PEPCK | **0.789** | -13.08 | 1.509 | 0.429 | 2.57 | 1.21 | 0.33 | 0.975 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r05/PEPCK/poses/chunk_487_lig_23_s1.dlg` | 4 |
| Capped | PptT | **0.672** | -16.04 | 0.763 | 0.500 | 2.84 | N/A | N/A | N/A | `exps_upperbound/acegen-reinforce/run_20260429_211516_r01/PptT/poses/chunk_0_lig_11_s1.dlg` | 6 |
| Capped | TTK | **0.837** | -11.88 | 1.017 | 0.653 | 2.30 | -8.89 | 0.25 | 0.994 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r04/TTK/poses/chunk_764_lig_47_s1.dlg` | 8 |
| Capped | VEGFR2 | **0.185** | -6.57 | 0.000 | 0.732 | 2.46 | -10.12 | 0.17 | 0.998 | `exps_upperbound/acegen-reinforce/run_20260429_211516_r01/VEGFR2/poses/chunk_254_lig_1_s0.dlg` | 4 |
| Uncapped | CHK1 | **0.789** | -12.60 | 1.151 | 0.657 | 2.55 | 1.11 | 0.33 | 0.832 | `exps/acegen-reinforce/run_20260430_220050_r05/CHK1/poses/chunk_699_lig_18_s0.dlg` | 7 |
| Uncapped | DPP4 | **0.907** | -11.89 | 1.237 | 0.918 | 3.41 | -19.12 | 0.00 | 0.794 | `exps/acegen-reinforce/run_20260430_220050_r01/DPP4/poses/chunk_128_lig_2_s0.dlg` | 9 |
| Uncapped | ITK | **0.881** | -10.55 | 1.188 | 0.782 | 2.29 | -5.05 | 0.17 | 0.998 | `exps/acegen-reinforce/run_20260430_220050_r05/ITK/poses/chunk_320_lig_14_s0.dlg` | 8 |
| Uncapped | PEPCK | **0.787** | -13.07 | 1.505 | 0.429 | 2.57 | 1.21 | 0.33 | 0.963 | `exps/acegen-reinforce/run_20260430_220050_r03/PEPCK/poses/chunk_60_lig_14_s1.dlg` | 3 |
| Uncapped | PptT | **0.672** | -16.03 | 0.762 | 0.500 | 2.84 | N/A | N/A | N/A | `exps/acegen-reinforce/run_20260430_220050_r02/PptT/poses/chunk_0_lig_11_s1.dlg` | 3 |
| Uncapped | TTK | **0.803** | -10.03 | 0.548 | 0.915 | 2.88 | -10.57 | 0.08 | 0.975 | `exps/acegen-reinforce/run_20260430_220050_r05/TTK/poses/chunk_574_lig_0_s0.dlg` | 4 |
| Uncapped | VEGFR2 | **0.597** | -9.97 | 0.143 | 0.863 | 4.67 | -4.33 | 0.00 | 0.987 | `exps/acegen-reinforce/run_20260430_220050_r04/VEGFR2/poses/chunk_759_lig_0_s0.dlg` | 2 |

## Model: REINVENT

| Experiment | Target | Geom. Mean Score | Docking Score (raw) | Normalized Docking | QED | SA | MolSkill | Stoplight | AIZynthFinder | Pose File (Relative Path) | Pose Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| Capped | CHK1 | **0.777** | -10.32 | 0.725 | 0.688 | 2.49 | -0.56 | 0.08 | 0.951 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r05/CHK1/poses/chunk_0_lig_15_s0.dlg` | 8 |
| Capped | DPP4 | **0.855** | -12.04 | 1.297 | 0.800 | 2.75 | -7.88 | 0.33 | 0.817 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r01/DPP4/poses/chunk_256_lig_47_s0.dlg` | 1 |
| Capped | ITK | **0.899** | -10.91 | 1.337 | 0.709 | 2.39 | -10.71 | 0.17 | 0.994 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r03/ITK/poses/chunk_768_lig_39_s0.dlg` | 5 |
| Capped | PEPCK | **0.786** | -13.04 | 1.493 | 0.429 | 2.57 | 1.21 | 0.33 | 0.963 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r03/PEPCK/poses/chunk_735_lig_58_s1.dlg` | 3 |
| Capped | PptT | **0.701** | -11.92 | 0.300 | 0.663 | 2.21 | -15.36 | 0.25 | 0.994 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r02/PptT/poses/chunk_765_lig_5_s1.dlg` | 9 |
| Capped | TTK | **0.811** | -10.79 | 0.740 | 0.700 | 2.55 | -15.61 | 0.25 | 0.952 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r01/TTK/poses/chunk_891_lig_49_s1.dlg` | 4 |
| Capped | VEGFR2 | **0.193** | -6.05 | 0.000 | 0.834 | 2.38 | -11.13 | 0.00 | 0.998 | `exps_upperbound/acegen-reinvent/run_20260429_234501_r05/VEGFR2/poses/chunk_624_lig_0_s0.dlg` | 3 |
| Uncapped | CHK1 | **0.774** | -9.49 | 0.570 | 0.723 | 2.30 | -8.26 | 0.17 | 0.963 | `exps/acegen-reinvent/run_20260430_220204_r01/CHK1/poses/chunk_632_lig_25_s0.dlg` | 9 |
| Uncapped | DPP4 | **0.894** | -11.80 | 1.201 | 0.897 | 2.87 | -12.59 | 0.08 | 0.832 | `exps/acegen-reinvent/run_20260430_220204_r02/DPP4/poses/chunk_832_lig_9_s1.dlg` | 4 |
| Uncapped | ITK | **0.888** | -10.69 | 1.246 | 0.751 | 2.14 | -13.29 | 0.42 | 0.994 | `exps/acegen-reinvent/run_20260430_220204_r01/ITK/poses/chunk_512_lig_1_s0.dlg` | 8 |
| Uncapped | PEPCK | **0.787** | -13.06 | 1.501 | 0.429 | 2.57 | 1.21 | 0.33 | 0.963 | `exps/acegen-reinvent/run_20260430_220204_r03/PEPCK/poses/chunk_60_lig_15_s1.dlg` | 4 |
| Uncapped | PptT | **0.702** | -12.35 | 0.348 | 0.631 | 2.38 | -10.27 | 0.17 | 0.975 | `exps/acegen-reinvent/run_20260430_220204_r05/PptT/poses/chunk_935_lig_25_s1.dlg` | 2 |
| Uncapped | TTK | **0.806** | -12.52 | 1.179 | 0.570 | 2.96 | -2.90 | 0.25 | 0.975 | `exps/acegen-reinvent/run_20260430_220204_r01/TTK/poses/chunk_893_lig_21_s1.dlg` | 6 |
| Uncapped | VEGFR2 | **0.697** | -12.97 | 1.197 | 0.398 | 4.12 | -7.46 | 0.58 | 0.762 | `exps/acegen-reinvent/run_20260430_220204_r01/VEGFR2/poses/chunk_573_lig_0_s0.dlg` | 5 |

## How to Retrieve the Poses

To grab any specific pose for visualization (e.g., in PyMOL or ChimeraX), locate the `.dlg` file listed in the table. 
The `.dlg` file contains the docking run outputs (in AutoDock format). The `Pose Index` indicates the zero-indexed conformation model number within the file.

### DLG to SDF Conversion Utility
You can also use the conversion script provided in `scripts/analysis/convert_dlg_to_sdf.py` to extract any conformation pose to a `.sdf` file:

```bash
python3 scripts/analysis/convert_dlg_to_sdf.py \
    -i <path_to_dlg_file> \
    -o <output_path_to_sdf> \
    -p <pose_index>
```
