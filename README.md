# ImportantDirections
**Importand Directions** is a scalable method to construct prediction intervals for large neural networks. The method works with a pretrained network, so you you can use your existing model, but the training set is also required. This repositiry contains a reference implementation of the method and related experiments and accompanies the corresponding [paper](#citation).

# Usage

In order to obtain prediction intervals with our method you will need a network that was trained for regression with the $MSE$ loss and $L_2$ regularization with parameter $\lambda$. You also need to have access to the original dataset that the model was trained on.

The only hyperparameter is `rank` that determines the rank of the underlying approximation to the covariance matrix.

```python
    imp_dirs = ImportantDirectionsPytorch(model, rank=max_rows, alpha_final=lamda)
    imp_dirs.fit(data_module.X_train, data_module.y_train)
    y_pred, y_l, y_u = imp_dirs.predict_to_numpy(data_module.X_test, data_module.y_test)
```

# Reproduce experiments

You can run the experiments from the paper by exicuting `bash run_all.sh`. This will produce `CSV` files with raw metrics, which then can be aggregated with `agg_results.py`.

# Citation
The original paper canbe found [here](https://link.springer.com/chapter/10.1007/978-3-031-16500-9_19). Preprint version is available [here](https://arxiv.org/abs/2205.03194). If you use our method, we kindly ask you to cite:

Fishkov, A., Panov, M. (2022). Scalable Computation of Prediction Intervals for Neural Networks via Matrix Sketching. In: , et al. Analysis of Images, Social Networks and Texts. AIST 2021. Lecture Notes in Computer Science, vol 13217. Springer, Cham. https://doi.org/10.1007/978-3-031-16500-9_19

```
@InProceedings{10.1007/978-3-031-16500-9_19,
author="Fishkov, Alexander
and Panov, Maxim",
title="Scalable Computation of Prediction Intervals for Neural Networks via Matrix Sketching",
booktitle="Analysis of Images, Social Networks and Texts",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="225--238",
isbn="978-3-031-16500-9"
}
```
