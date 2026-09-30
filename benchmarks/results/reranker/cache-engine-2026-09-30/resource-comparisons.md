# Complete diagnostic resource comparisons

Each row derives from the actual call records. Wall values are microseconds; HWM is KiB; CPU values are100Hz ticks. Counts disclose process replicas. N/A percent means zero baseline. See README scope before interpreting these figures.

## q80 / A / cold

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2.56898e+06 | 2.46139e+06 | -107591 | -4.19% |
| min_wall_us | 2.53463e+06 | 2.46124e+06 | -73390 | -2.90% |
| max_wall_us | 2.60333e+06 | 2.46154e+06 | -141792 | -5.45% |
| median_encode_us | 4055 | 4118 | +63 | +1.55% |
| fresh_scores | 60 | 60 | +0 | +0.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 1019 | 940 | -79 | -7.75% |
| total_process_system_ticks | 87 | 75 | -12 | -13.79% |
| max_process_hwm_kib | 1.12825e+06 | 1.12866e+06 | +408 | +0.04% |

## q80 / A / hot

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 20 | 20 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 2750 | 4133.5 | +1383.5 | +50.31% |
| min_wall_us | 2396 | 3607 | +1211 | +50.54% |
| max_wall_us | 4432 | 5772 | +1340 | +30.23% |
| median_encode_us | 0 | 1042.5 | +1042.5 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 600 | 600 | +0 | +0.00% |
| total_process_user_ticks | 6 | 12 | +6 | +100.00% |
| total_process_system_ticks | 0 | 1 | +1 | N/A |
| max_process_hwm_kib | 1.12826e+06 | 1.12867e+06 | +408 | +0.04% |

## q80 / A / off

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 1 | 1 | +0 | +0.00% |
| median_wall_us | 724604 | 722368 | -2236 | -0.31% |
| min_wall_us | 3679 | 4333 | +654 | +17.78% |
| max_wall_us | 1.44553e+06 | 1.4404e+06 | -5126 | -0.35% |
| median_encode_us | 0 | 0 | +0 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 0 | 0 | +0 | N/A |
| total_process_user_ticks | 126 | 121 | -5 | -3.97% |
| total_process_system_ticks | 23 | 27 | +4 | +17.39% |
| max_process_hwm_kib | 760828 | 760056 | -772 | -0.10% |

## q80 / A / unchanged-call

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2850 | 4111 | +1261 | +44.25% |
| min_wall_us | 2775 | 3975 | +1200 | +43.24% |
| max_wall_us | 2925 | 4247 | +1322 | +45.20% |
| median_encode_us | 0 | 1051.5 | +1051.5 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 0 | 0 | +0 | N/A |
| total_process_system_ticks | 0 | 0 | +0 | N/A |
| max_process_hwm_kib | 1.12824e+06 | 1.12847e+06 | +232 | +0.02% |

## q80 / B / cold

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2.48879e+06 | 2.62252e+06 | +133735 | +5.37% |
| min_wall_us | 2.47612e+06 | 2.5972e+06 | +121078 | +4.89% |
| max_wall_us | 2.50146e+06 | 2.64785e+06 | +146392 | +5.85% |
| median_encode_us | 4149 | 3951 | -198 | -4.77% |
| fresh_scores | 60 | 60 | +0 | +0.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 972 | 1051 | +79 | +8.13% |
| total_process_system_ticks | 81 | 72 | -9 | -11.11% |
| max_process_hwm_kib | 1.12823e+06 | 1.12846e+06 | +228 | +0.02% |

## q80 / B / hot

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 20 | 20 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 2874 | 4487.5 | +1613.5 | +56.14% |
| min_wall_us | 2462 | 3645 | +1183 | +48.05% |
| max_wall_us | 5072 | 6388 | +1316 | +25.95% |
| median_encode_us | 0 | 1178 | +1178 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 600 | 600 | +0 | +0.00% |
| total_process_user_ticks | 4 | 7 | +3 | +75.00% |
| total_process_system_ticks | 0 | 1 | +1 | N/A |
| max_process_hwm_kib | 1.12826e+06 | 1.12867e+06 | +408 | +0.04% |

## q80 / B / unchanged-call

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2745.5 | 3894 | +1148.5 | +41.83% |
| min_wall_us | 2712 | 3747 | +1035 | +38.16% |
| max_wall_us | 2779 | 4041 | +1262 | +45.41% |
| median_encode_us | 0 | 1059.5 | +1059.5 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 1 | 0 | -1 | -100.00% |
| total_process_system_ticks | 0 | 0 | +0 | N/A |
| max_process_hwm_kib | 1.12826e+06 | 1.12867e+06 | +408 | +0.04% |

## q81 / N / new-query

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 4 | 4 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 1.20554e+06 | 1.2395e+06 | +33960.5 | +2.82% |
| min_wall_us | 1.11747e+06 | 1.14079e+06 | +23316 | +2.09% |
| max_wall_us | 1.38998e+06 | 1.3667e+06 | -23287 | -1.68% |
| median_encode_us | 2924 | 3117.5 | +193.5 | +6.62% |
| fresh_scores | 120 | 120 | +0 | +0.00% |
| actual_offered | 120 | 120 | +0 | +0.00% |
| total_process_user_ticks | 1463 | 1517 | +54 | +3.69% |
| total_process_system_ticks | 44 | 37 | -7 | -15.91% |
| max_process_hwm_kib | 1.13695e+06 | 1.13805e+06 | +1096 | +0.10% |

## q81 / N / off

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 1 | 1 | +0 | +0.00% |
| processes | 1 | 1 | +0 | +0.00% |
| median_wall_us | 31055 | 33474 | +2419 | +7.79% |
| min_wall_us | 31055 | 33474 | +2419 | +7.79% |
| max_wall_us | 31055 | 33474 | +2419 | +7.79% |
| median_encode_us | 0 | 0 | +0 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 0 | 0 | +0 | N/A |
| total_process_user_ticks | 6 | 7 | +1 | +16.67% |
| total_process_system_ticks | 1 | 0 | -1 | -100.00% |
| max_process_hwm_kib | 761184 | 760348 | -836 | -0.11% |

## q101 / A / changed-call

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 43788 | 579004 | +535216 | +1222.29% |
| min_wall_us | 41293 | 569567 | +528274 | +1279.33% |
| max_wall_us | 46283 | 588440 | +542157 | +1171.40% |
| median_encode_us | 269 | 1114.5 | +845.5 | +314.31% |
| fresh_scores | 2 | 40 | +38 | +1900.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 22 | 359 | +337 | +1531.82% |
| total_process_system_ticks | 2 | 4 | +2 | +100.00% |
| max_process_hwm_kib | 1.1273e+06 | 1.12827e+06 | +968 | +0.09% |

## q101 / A / cold

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2.20652e+06 | 2.59563e+06 | +389107 | +17.63% |
| min_wall_us | 2.20173e+06 | 2.46554e+06 | +263805 | +11.98% |
| max_wall_us | 2.21132e+06 | 2.72572e+06 | +514409 | +23.26% |
| median_encode_us | 3778 | 3887 | +109 | +2.89% |
| fresh_scores | 60 | 60 | +0 | +0.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 839 | 923 | +84 | +10.01% |
| total_process_system_ticks | 76 | 68 | -8 | -10.53% |
| max_process_hwm_kib | 1.12762e+06 | 1.12613e+06 | -1492 | -0.13% |

## q101 / A / hot

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 20 | 20 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 2621.5 | 4137 | +1515.5 | +57.81% |
| min_wall_us | 2434 | 3403 | +969 | +39.81% |
| max_wall_us | 5893 | 6515 | +622 | +10.55% |
| median_encode_us | 0 | 1101.5 | +1101.5 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 600 | 600 | +0 | +0.00% |
| total_process_user_ticks | 6 | 6 | +0 | +0.00% |
| total_process_system_ticks | 0 | 0 | +0 | N/A |
| max_process_hwm_kib | 1.12763e+06 | 1.12827e+06 | +644 | +0.06% |

## q101 / B / changed-call

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 36692.5 | 707009 | +670316 | +1826.85% |
| min_wall_us | 35935 | 696265 | +660330 | +1837.57% |
| max_wall_us | 37450 | 717753 | +680303 | +1816.56% |
| median_encode_us | 246.5 | 1192 | +945.5 | +383.57% |
| fresh_scores | 2 | 40 | +38 | +1900.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 19 | 413 | +394 | +2073.68% |
| total_process_system_ticks | 2 | 10 | +8 | +400.00% |
| max_process_hwm_kib | 1.12808e+06 | 1.12658e+06 | -1492 | -0.13% |

## q101 / B / cold

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 2 | 2 | +0 | +0.00% |
| processes | 2 | 2 | +0 | +0.00% |
| median_wall_us | 2.45534e+06 | 2.51267e+06 | +57332 | +2.33% |
| min_wall_us | 2.43759e+06 | 2.46402e+06 | +26428 | +1.08% |
| max_wall_us | 2.47309e+06 | 2.56133e+06 | +88236 | +3.57% |
| median_encode_us | 3563 | 3810.5 | +247.5 | +6.95% |
| fresh_scores | 60 | 60 | +0 | +0.00% |
| actual_offered | 60 | 60 | +0 | +0.00% |
| total_process_user_ticks | 963 | 916 | -47 | -4.88% |
| total_process_system_ticks | 82 | 79 | -3 | -3.66% |
| max_process_hwm_kib | 1.12677e+06 | 1.12775e+06 | +980 | +0.09% |

## q101 / B / hot

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 20 | 20 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 2839.5 | 3960.5 | +1121 | +39.48% |
| min_wall_us | 2323 | 3407 | +1084 | +46.66% |
| max_wall_us | 4196 | 5507 | +1311 | +31.24% |
| median_encode_us | 0 | 1004 | +1004 | N/A |
| fresh_scores | 0 | 0 | +0 | N/A |
| actual_offered | 600 | 600 | +0 | +0.00% |
| total_process_user_ticks | 6 | 4 | -2 | -33.33% |
| total_process_system_ticks | 0 | 1 | +1 | N/A |
| max_process_hwm_kib | 1.12808e+06 | 1.12775e+06 | -324 | -0.03% |

## q102 / N / new-query

| Metric | Before | After | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| calls | 4 | 4 | +0 | +0.00% |
| processes | 4 | 4 | +0 | +0.00% |
| median_wall_us | 1.13731e+06 | 1.30117e+06 | +163860 | +14.41% |
| min_wall_us | 1.05197e+06 | 1.17521e+06 | +123240 | +11.72% |
| max_wall_us | 1.18429e+06 | 1.35836e+06 | +174066 | +14.70% |
| median_encode_us | 2853 | 2767.5 | -85.5 | -3.00% |
| fresh_scores | 120 | 120 | +0 | +0.00% |
| actual_offered | 120 | 120 | +0 | +0.00% |
| total_process_user_ticks | 1357 | 1526 | +169 | +12.45% |
| total_process_system_ticks | 33 | 34 | +1 | +3.03% |
| max_process_hwm_kib | 1.13786e+06 | 1.13577e+06 | -2092 | -0.18% |
