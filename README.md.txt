## Meta-GTMP: a few-shot GNN-Transformer approach for Ames mutagenicity prediction

In this work, we introduce a few-shot GNN-Transformer, Meta-GTMP to predict the mutagenicity of a small amount of labeled molecules based on the results achieved for each individual strain involved in the Ames test. A two-module multi-task meta-learning framework combines the information of multiple mutagenic properties across few-shot tasks to leverage the complementarity among individual strains to model the overall Ames mutagenicity test result with limited available data. 

First, a GNN treats molecules as a set of node and edge features converted into graph embeddings by neighborhood aggregation. Then, a Transformer encoder preserves the global information in these vectorial embeddings to propagate deep representations across attention layers. The long-range dependencies captured express the global-semantic structure of molecular embeddings as a function to predict mutagenic properties in small drug repositories. It is demonstrated that this hybrid approach achieves a superior performance when predicting the overall Ames mutagenicity result over standard graph-based methods.

![ScreenShot](figures/ames-fs-gnntr.png?raw=true)

To address the challenge of low-data learning, we introduce a two-module meta-learning framework to quickly update model parameters across few-shot tasks for each individual strain to predict the overall Ames result in the test data.

![ScreenShot](figures/meta.png?raw=true)

Experiments on multi-property prediction data show that the proposed model significantly outperforms other simpler graph-based baselines in multi-task Ames mutagenicity prediction.

This repository provides the source code and datasets for the proposed work.

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Data Availability and Pre-Processing

The dataset, which includes information on the results of the Ames mutagencity test on various strains of *S. typhimurium* for 7367 compounds, was collected and organized by the [*Istituto Superiore di Sanità (ISS)*](http://https://www.iss.it/isstox).

Data is pre-processed and transformed into molecular graphs using RDKit.Chem. 

Data pre-processing and pre-trained models are implemented based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Code Usage

### Installation
We used the following Python packages for core development. We tested on Python 3.7.

```
- torch = 1.10.1
- torch-cluster = 1.5.9
- torch-geometric = 2.0.4
- torch-scatter = 2.0.9
- torch-sparse = 0.6.12
- torch-spline-conv = 1.2.1
- torchvision = 0.10.0
- vit-pytorch = 0.35.8
- scikit-learn = 1.0.2
- seaborn = 0.11.2
- scipy = 1.8.0
- numpy = 1.21.5
- tqdm = 4.50.0
- tensorflow = 2.8.0
- keras = 2.8.0
- tsnecuda = 3.0.1
- tqdm = 4.62.3
- matplotlib = 3.5.1
- pandas = 1.4.1
- networkx = 2.7.1
- rdkit
- bioalerts
```

## References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.: Strategies for pre-training graph neural networks. CoRR abs/1905.12265 (2020). https://doi.org/10.48550/ARXIV.1905.12265

```
@inproceedings{hu2020pretraining,
  title={Strategies for Pre-training Graph Neural Networks},
  author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=HJlWWJSFDH}
}

```
[2] Romualdo Benigni, Chiara Laura Battistelli, Cecilia Bossa,
Olga Tcheremenskaia, and Pierre Crettaz. Istituto
superiore di sanit`a: Isstox chemical toxicity databases,
2019. https://doi.org/10.1093/mutage/get016

```
@article{Benigni2019,
   author = {Romualdo Benigni and Chiara Laura Battistelli and Cecilia Bossa and Olga Tcheremenskaia and Pierre Crettaz},
   url = {https://www.iss.it/isstox},
   title = {Istituto Superiore di Sanità: ISSTOX Chemical Toxicity Databases},
   year = {2019},
}

```
[3] Vision Transformers with PyTorch. https://github.com/lucidrains/vit-pytorch

```
@misc{Phil Wang,
  author = {Phil Wang},
  title = {Vision Transformers},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lucidrains/vit-pytorch}}
}


```
[4] Bioalerts: A python library for the derivation of structural alerts from bioactivity and toxicity data sets.

```
@misc{Isidro Cortes-Ciriano2016,
   author = {Isidro Cortes-Ciriano},
   journal = {Journal of Cheminformatics},
   month = {3},
   title = {Bioalerts: A python library for the derivation of structural alerts from bioactivity and toxicity data sets},
   doi = {10.1186/s13321-016-0125-7},
   year = {2016},
   publisher = {BioMed Central Ltd.},
   journal = {Journal of Cheminformatics},
  }
}

```
