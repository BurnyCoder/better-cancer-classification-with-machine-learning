| Arm | Backbone(s) | Head | Dim | OOF BA [95% CI] | Valid BA [95% CI] | Valid BA seed SD | Valid AUROC | Publishable |
|---|---|---|---|---|---|---|---|---|
| `control_color_histogram` | — | logreg | 96 | 0.7870 [0.7580, 0.8106] | 0.8098 [0.7530, 0.8472] | 0.0000 | 0.8923 | yes |
| `control_majority` | — | — | — | — | 0.5000 [0.5000, 0.5000] | 0.0000 | 0.5000 | yes |
| `control_shuffled_labels` | Hibou-L | linear | 1024 | 0.5204 [0.4935, 0.5457] | 0.4731 [0.4280, 0.5212] | 0.0471 | 0.4637 | yes |
| `S0_eva_exact_hibou_l` | Hibou-L | linear | 1024 | — | 0.9396 [0.8936, 0.9638] | 0.0007 | 0.9833 | yes |
| `S0_hibou_l_cls_linear` | Hibou-L | linear | 1024 | 0.9498 [0.9366, 0.9600] | 0.9367 [0.8888, 0.9617] | 0.0015 | 0.9823 | yes |
| `S1_b_cls24_mean24` | Hibou-L | linear | 2048 | 0.9491 [0.9353, 0.9598] | 0.9353 [0.8830, 0.9607] | 0.0019 | 0.9816 | yes |
| `S1_c_dinov2_last4` | Hibou-L | linear | 5120 | 0.9457 [0.9303, 0.9573] | 0.9316 [0.8712, 0.9593] | 0.0025 | 0.9796 | yes |
