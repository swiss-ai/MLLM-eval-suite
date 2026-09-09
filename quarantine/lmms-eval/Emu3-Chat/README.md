Emu3-Chat July-13 cells with mixed skipped/generated chunks — DO NOT PUBLISH.

The emu chat models appended skip placeholders during the scan loop but
generated answers after the batch, so any chunk containing both associated
answers with the wrong requests (fixed 2026-08-14, lmms-eval commit
e227860b "emu chat models: place answers by sample index").

Quarantined tasks (skipped/total): mmmu_val 39/900, mmmu_pro 136/3460,
scienceqa 2224/4241, frieda 296/500, medxpertqa_mm 419/2000,
iconqa_val 11534/21488.

Fully-skipped tasks (rsrcc, muirbench, medqa, medmcqa, pubmedqa,
medxpertqa_text, geobench_temporal — N/N placeholders, no interleave
possible) were left in results/: they reflect the skip policy, not the bug.

Rerun on the fixed code before these six cells return to the dashboard.
