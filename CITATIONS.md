# Citations

References for the methods, models, and datasets used in this project.

## Benchmark / foundational work

1. C. J. Bartel, A. Trewartha, Q. Wang, A. Dunn, A. Jain, G. Ceder.
   *A critical examination of compound stability predictions from machine-learned formation energies.*
   npj Computational Materials **6**, 97 (2020).
   DOI: [10.1038/s41524-020-00362-y](https://doi.org/10.1038/s41524-020-00362-y) - arXiv: 2001.10591
   The benchmark this work extends; the 2020 MP hull data (`data/2020/`) and baseline model predictions (`data/2020/ml/`) are derived from the TestStabilityML repository.

## Dataset

2. A. Jain et al. *The Materials Project: A materials genome approach to accelerating materials innovation.*
   APL Materials **1**, 011002 (2013). DOI: [10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

## Models and descriptors

3. L. Ward, A. Agrawal, A. Choudhary, C. Wolverton. *A general-purpose machine learning framework for predicting properties of inorganic materials* (Magpie).
   npj Computational Materials **2**, 16028 (2016). DOI: [10.1038/npjcompumats.2016.28](https://doi.org/10.1038/npjcompumats.2016.28)

4. B. Meredig et al. *Combinatorial screening for new materials in unconstrained composition space with machine learning* (Meredig features).
   Physical Review B **89**, 094104 (2014). DOI: [10.1103/PhysRevB.89.094104](https://doi.org/10.1103/PhysRevB.89.094104)

5. D. Jha et al. *ElemNet: Deep learning the chemistry of materials from only elemental composition.*
   Scientific Reports **8**, 17593 (2018). DOI: [10.1038/s41598-018-35934-y](https://doi.org/10.1038/s41598-018-35934-y)

6. R. E. A. Goodall, A. A. Lee. *Predicting materials properties without crystal structure: deep representation learning from stoichiometry* (Roost).
   Nature Communications **11**, 6280 (2020). DOI: [10.1038/s41467-020-19964-1](https://doi.org/10.1038/s41467-020-19964-1)

7. T. Xie, J. C. Grossman. *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties* (CGCNN).
   Physical Review Letters **120**, 145301 (2018). DOI: [10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)

8. B. Deng et al. *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.*
   Nature Machine Intelligence **5**, 1031 (2023). DOI: [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)

9. K. Choudhary, B. DeCost. *Atomistic Line Graph Neural Network for improved materials property prediction* (ALIGNN).
   npj Computational Materials **7**, 185 (2021). DOI: [10.1038/s41524-021-00650-1](https://doi.org/10.1038/s41524-021-00650-1)

10. A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain. *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.*
    npj Computational Materials **6**, 138 (2020). DOI: [10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3)

## Software

11. L. Ward et al. *Matminer: An open source toolkit for materials data mining.*
    Computational Materials Science **152**, 60-69 (2018). DOI: [10.1016/j.commatsci.2018.05.018](https://doi.org/10.1016/j.commatsci.2018.05.018)

12. S. P. Ong et al. *Python Materials Genomics (pymatgen).*
    Computational Materials Science **68**, 314-319 (2013). DOI: [10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028)

13. F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR **12**, 2825 (2011).

14. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

## Training methods

15. T. Yu et al. *Gradient Surgery for Multi-Task Learning* (PCGrad). NeurIPS 2020. arXiv: 2001.06782

16. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv: 1412.6980

---

A fuller annotated reference list (including theoretical background for the interference index,
multi-task loss balancing, and physical context) is preserved in the project's working notes.
