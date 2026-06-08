
SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks
  ---

  ## Requirements

  Python >= 3.8
  PyTorch >= 1.9.0
  torchvision >= 0.10.0
  timm == 0.4.12

  Install all dependencies:

  ```bash
  pip install -r requirements.txt
  ```
  ---
  Dataset Setup

  Edit config.py to point each dataset variable to your local copy:

  ┌───────────────────────────────────────┬─────────────────────────────────────┐
  │               Variable                │               Dataset               │
  ├───────────────────────────────────────┼─────────────────────────────────────┤                       │
  │ aircraft_root                         │ FGVC-Aircraft (fgvc-aircraft-2013b) │
  ├───────────────────────────────────────┼─────────────────────────────────────┤
  │ cub_root                              │ CUB-200-2011                        │
  ├───────────────────────────────────────┼─────────────────────────────────────┤
  │ car_root                              │ Stanford Cars                       │
  ├───────────────────────────────────────┼─────────────────────────────────────┤    

  Pre-processed OSR splits are included in data/open_set_splits/.

  ---
  Training

  Training uses a two-stage pipeline.

  Stage 1 — Feature learning with SpHOR

  sh bash_scripts/osr_stage1.sh

  This trains a ResNet-50 backbone on the target dataset using the SpHOR. At the end of training, note the
  experiment ID printed to the console:

  Experiment saved to: savedir/methods/ARPL/log/(DD.MM.YYYY_|_SS.mmm)

  Stage 2 — Classifier fine-tuning and evaluation

  Open bash_scripts/osr_stage2_plus_testing.sh and set --exp_id to the ID  
  printed by Stage 1:

  --exp_id="(DD.MM.YYYY_|_SS.mmm)"   

  Then run:

  sh bash_scripts/osr_stage2_plus_testing.sh

  Stage 2 loads the Stage 1 checkpoint, trains a linear classifier on frozen
  features, then evaluates open-set performance (AUROC, ACC, OSCR) under both
  MSP, NNGuide, PostMAX and k-NN scoring functions across Easy and Hard open-set splits for SPHOR


  ---
  Repository Structure

  ├── bash_scripts/
  │   ├── osr_stage1.sh              # Stage 1 training
  │   └── osr_stage2_plus_testing.sh # Stage 2 fine-tuning + evaluation
  ├── config.py                      # Dataset and path configuration
  ├── data/
  │   ├── open_set_splits/           # Pre-computed OSR class splits
  │   └── *.py                       # Dataset loaders
  ├── methods/
  │   └── ARPL/
  │       ├── core/                  # Training and test loops
  │       ├── loss/
  │       │   └──sphor.py   # Loss (main contributions)
  │       ├── arpl_models/           # Model architectures
  │       ├── osr_firststage.py             # Stage 1 entry point
  │       └── osr_classifier.py      # Stage 2 entry point
  ├── models/
  │   └── model_utils.py             # Backbone loading utilities
  ├── utils/
  │   ├── schedulers.py              # LR schedulers
  │   └── utils.py                   # Experiment init, seeding, metrics
  ├── requirements.txt
  └── savedir/                       # Experiment outputs (checkpoints, logs)

  ---
  Evaluation Metrics

  ┌─────────┬─────────────────────────────────────────────┐
  │ Metric  │                 Description                 │
  ├─────────┼─────────────────────────────────────────────┤
  │ ACC     │ Closed-set top-1 accuracy                   │
  ├─────────┼─────────────────────────────────────────────┤
  │ AUROC   │ Area under ROC for open-set detection       │
  ├─────────┼─────────────────────────────────────────────┤  
  │ OSCR    │ Open-Set Classification Rate                │
  ├─────────┼─────────────────────────────────────────────┤

  ---
  Supported Datasets

  Fine-grained: CUB-200-2011, Stanford Cars, FGVC-Aircraft, PKU-Aircraft

@InProceedings{Bahavan_2026_CVPR,
    author    = {Bahavan, Thiru Thillai Nadarasar and Seneviratne, Sachith and Halgamuge, Saman},
    title     = {SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6901-6910}
}

  Code based on ARPL(https://github.com/gary23ai/ARPL) and SSB (https://github.com/sgvaze/osr_closed_set_all_you_need) 
