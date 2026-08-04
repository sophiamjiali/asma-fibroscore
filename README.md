# αSMAFibroScore
Automated quantification of αSMA-stained bone marrow biopsy whole-slide images for CHIP research, designed to integrate with QuPath.

## What it does
 
Given an αSMA IHC whole-slide image, this tool:
- Segments marrow vasculature via a patch-based U-Net
- Computes patient-level fibroblast/vessel density (αSMA+ area fraction)
- Predicts CHIP status from the derived density score
- Generates fibroblast and vessel overlay images for QuPath visualization
## Scope
 
This is a frozen, deployment-only inference tool. Model training, hyperparameter optimization, and validation experiments are maintained separately in [`chip-stroma-analysis`](#) (the research repository this tool was distilled from).
 
## Outputs
 
| Output | Description |
|---|---|
| Fibroblast density score | αSMA+ area / tissue area |
| CHIP status prediction | Binary classification from density score |
| Fibroblast overlay | Per-patch αSMA+ segmentation mask |
| Vessel overlay | Detected vessel structures |
 
## Usage
 
```bash
python run_inference.py --slide path/to/slide.svs --output results/
```
 
## Requirements
 
See `requirements.txt`. Model weights are loaded from a fixed, versioned checkpoint (finalized all-patient config).
