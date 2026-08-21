# Publication scope and reproduction boundary

This repository contains a code reproduction of the WirelessJEPA training/evaluation pipeline. The paper PDF, RML2016.10a dataset, checkpoints and generated outputs are intentionally excluded.

RML2016.10a is single-antenna data expanded to the model input grid. The recorded results validate the software pipeline and mask-geometry comparison, not a strict reproduction of the paper's real multi-antenna pretraining setting. See `REPRODUCTION_REPORT.md` for the full limitation analysis.

## Portfolio-preparation check (2026-08-21)

A fresh run in the available `d2l` environment passed preprocessing, all four masking strategies, CUDA device/tensor operations, model creation/forward pass, and the expected no-data loading branch. The combined script reported 5/7 because that environment lacks `seaborn` and the managed sandbox denied creation of its specifically named `test_outputs` directory. The historical 7/7 result in the reproduction report was not overwritten; full training and evaluation were not rerun.
