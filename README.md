## Meta-GTMP: a few-shot GNN-Transformer approach for Ames mutagenicity prediction

In this work, we introduce a few-shot GNN-Transformer, Meta-GTMP to predict the mutagenicity of a small amount of labeled molecules based on the results achieved for each individual strain involved in the Ames test. A two-module multi-task meta-learning framework combines the information of multiple mutagenic properties across few-shot tasks to leverage the complementarity among individual strains to model the overall Ames mutagenicity test result with limited available data. 

First, a GNN treats molecules as a set of node and edge features converted into graph embeddings by neighborhood aggregation. Then, a Transformer encoder preserves the global information in these vectorial embeddings to propagate deep representations across attention layers. The long-range dependencies captured express the global-semantic structure of molecular embeddings as a function to predict mutagenic properties in small drug repositories. It is demonstrated that this hybrid approach achieves a superior performance when predicting the overall Ames mutagenicity result over the standard graph-based methods.

![ScreenShot](figures/gnntr.png?raw=true)

To address the challenge of low-data, we introduce a two-module meta-learning framework to quickly update model parameters across few-shot tasks for each individual bacterail strain involved in the Ames test to predict the overall Ames result with just a few labeled compounds. Few-shot experiments in the 5-shot and 10-shot settings show that the proposed Meta-GTMP model significantly outperforms the graph-based baselines in Ames mutagenicity prediction.

![ScreenShot](figures/meta.png?raw=true)

Laboratory experiments are conducted using six selected compounds with a diverse range of chemical structures and unknown mutagenicity labels. This experimental validation step confirmed that Meta-GTMP effectively recognizes the mutagenic and non-mutagenic samples, while achieving interpretable results using a node-edge attribute masking strategy. In this interpretability study, we compute a set of node and edge scores for each atom and chemical bond in a molecule, providing important insights into the key chemical substructures influencing mutagenicity. These findings support the superior utility of the Meta-GTMP framework for drug discovery and mutagenicity assessments.

![ScreenShot](figures/node-edge-mask.png?raw=true)

This repository provides the source code and datasets for the proposed work.

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Data Availability and Pre-Processing

The dataset, which includes information on the results of the Ames mutagencity test on various strains of *S. typhimurium* for 7367 compounds, was collected and organized by the [*Istituto Superiore di Sanità (ISS)*](http://https://www.iss.it/isstox).

Data is pre-processed and transformed into molecular graphs using RDKit.Chem. 

The implementation and pre-trained models are based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Python Packages

We used the following Python packages for core development. We tested on Python 3.9.

```
- torch = 1.13.0+cu116 
- torch-cluster = 1.6.1
- torch-geometric =  2.3.0
- torch-scatter = 2.1.1
- torch-sparse = 0.6.17
- torch-spline-conv =  1.2.2
- torchvision = 0.14.0
- scikit-learn = 1.2.2
- seaborn = 0.12.2
- scipy = 1.7.3
- numpy = 1.22.3
- tqdm = 4.65.0
- tsnecuda = 3.0.1
- tqdm = 4.65.0
- matplotlib = 3.4.3 
- pandas = 1.5.3 
- networkx =  3.1.0
- rdkit = 2022.03.5

```

## Running the Code

This section provides an overview of the main scripts included in the repository and instructions for executing them.

### Meta-GTMP (Few-Shot GNN-Transformer):

gnntr_train.py: Performs meta-training using 5-shot or 10-shot support sets. This trains the GNN + Transformer model on strain-specific few-shot tasks using the ISSSTY dataset.

gnntr_test.py: Performs meta-testing to evaluate the trained model on the overall Ames mutagenicity classification task.

gnntr_eval.py: Evaluates the model’s performance on disjoint query sets, computing metrics such as ROC-AUC, accuracy, sensitivity, precision, and F1-score.

gnntr_mask.py: Executes the interpretability pipeline using the node-edge masking strategy to highlight substructures most relevant to mutagenicity or non-mutagenicity.

mask_model.py: Defines the masking model used for interpretability analysis in conjunction with gnntr_mask.py.

### Classical Machine Learning Baselines:

ml-baselines.py: Trains and evaluates baseline machine learning models (Random Forest, SVM, KNN, Gaussian Process) using molecular fingerprints (ECFP4 and MACCS) for overall Ames label prediction. Used for performance comparison with the proposed model.

### Utilities and Helper Scripts:

data.py: Handles data loading and preprocessing. Converts SMILES into molecular graphs using RDKit and PyTorch Geometric.

gnn_models.py: Defines several GNN architectures including Graph Isomorphism Networks (GIN).

gnn_tr.py: Defines the hybrid GNN + Transformer model architecture used in the Meta-GTMP framework.

train_model.py and test_model.py: Main scripts used to run the previous scripts gnntr_train.py, gnntr_test.py for training and evaluation.

eval_model.py: Main script to run gnntr_eval.py for computing evaluation metrics across experiments with the trained models.

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

[2] Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: 34th International Conference on Machine Learning, ICML 2017, vol. 3 (2017). https://doi.org/10.48550/arXiv.1703.03400

```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

[3] Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., & Chawla, N. V. (2021). Few-shot graph learning for molecular property prediction. In The Web Conference 2021 - Proceedings of the World Wide Web Conference, WWW 2021 (pp. 2559–2567). Association for Computing Machinery, Inc. https://doi.org/10.1145/3442381.3450112
```
@article{guo2021few,
  title={Few-Shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2102.07916},
  year={2021}
}
```

[4] Romualdo Benigni, Chiara Laura Battistelli, Cecilia Bossa,
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

[5] Vision Transformers with PyTorch. https://github.com/lucidrains/vit-pytorch

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
