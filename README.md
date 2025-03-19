# SphOR:A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Learning Models. (Arxiv)
Official PyTorch implementation of ["**SphOR:A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Learning Models.**"](https://scholar.google.com.au/citations?view_op=view_citation&hl=en&user=9cafqywAAAAJ&sortby=pubdate&citation_for_view=9cafqywAAAAJ:m4fbC6XIj1kC), [Nadarasar Bahavan](https://scholar.google.com/citations?user=AW1LjIkAAAAJ&hl=en), [Sachith Seneviratne](https://scholar.google.com/citations?user=nvv8iZEAAAAJ&hl=en), and [Saman K. Halgamuge](https://scholar.google.com.au/citations?user=9cafqywAAAAJ&hl=en).

> **Abstract:** *The widespread use of deep learning classifiers necessitates Open-set recognition (OSR), which enables the identification of input data not only from classes known during training but also from unknown classes that might be present in test data. Many existing OSR methods are computationally expensive due to the reliance on complex generative models or suffer from high training costs. We investigate OSR from a representation-learning perspective, specifically through spherical embeddings. We introduce SphOR, a computationally efficient representation learning method that models the feature space as a mixture of von Mises-Fisher distributions. This approach enables the use of semantically ambiguous samples during training, to improve the detection of samples from unknown classes. We further explore the relationship between OSR performance and key representation learning properties which influence how well features are structured in high-dimensional space. Extensive experiments on multiple OSR benchmarks demonstrate the effectiveness of our method, producing state-of-the-art results, with improvements up-to 6% that validate its performance.*

<p align="center">
    <img src=./img/overview.jpg width="800">
</p>

## 1. Requirements
### Environments
We used Conda for the environments, please refer to the generated environment YML file for more details

### Datasets
For Tiny-ImageNet, please download the following datasets to ```./data/tiny_imagenet```.
-   [tiny_imagenet](https://drive.google.com/file/d/1vR8ltP_U0UCM42pqz8q4mTbXcvipNNWP/view?usp=sharing)

## 2. Training & Evaluation

### Benchmark 01 : Unknown Detection (Benchmark from Neal et. al: Open set learning with counter-factual images)
Set each dataset to the correct configs, use ID=0..4 for each correspdoning dataset split, and average the AUROC scores manually. Use the model_suffix to track the corresponding dataset/edge case.
```train
python tin_stageone_training.py --id $ID --model_suffix $model_suffix
python tin_stagetwo_testing.py --id $ID --model_suffix $model_suffix
```

### Benchmark 02 : MNIST Openset  (Benchmark from Zhou et. al: Learning placeholders for open-set recognition. )
Set each dataset to the correct configs, use ID=5, model_suffix=main.pt (trained)
```train
python tin_stageone_training.py --id $ID --model_suffix $model_suffix
python mnistplus_testing.py --id $ID --model_suffix $model_suffix
```
### Benchmark 03 : CIFAR10 Openset (Benchmark from Zhou et. al: Learning placeholders for open-set recognition. )
Set each dataset to the correct configs, use ID=5 , model_suffix=main.pt (trained)
```train
python tin_stageone_training.py --id $ID --model_suffix $model_suffix
python cifarplus_testing.py --id $ID --model_suffix $model_suffix
```

### Benchmark 04 : CIFAR10 NearOOD/FarOOD (Benchmark from Chen et. al: Adversarial reciprocal points learning for open set recognition. )
Set each dataset to the correct configs, use ID=5 , model_suffix=main.pt (trained)
```train
python tin_stageone_training.py --id $ID --model_suffix $model_suffix
python cifarood_testing.py --id $ID --model_suffix $model_suffix
```

<!-- 
## 3. Results
### We visualize the deep feature of Softmax/GCPL/ARPL/ARPL+CS as below.

<!-- <!-- <p align="center">
    <img src=./img/results.jpg width="800">
</p>

> Colored triangles represent the learned reciprocal points of different known classes. -->


## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@misc{bahavan2025sphorrepresentationlearningperspective,
      title={SphOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Learning Models}, 
      author={Nadarasar Bahavan and Sachith Seneviratne and Saman Halgamuge},
      year={2025},
      eprint={2503.08049},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.08049}, 
}
```
## References
We thank the authors of the following repositories for their excellent codebase providing reusable functions:
[ConOSR](https://github.com/NJU-RINC/ConOSR)
[ARPL](https://github.com/iCGY96/ARPL)