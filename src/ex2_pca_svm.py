"""Use PCA to improve soft-margin SVM for face recognition."""

from collections.abc import Callable
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from util_pca import plot_expl_var, plot_image_matrix, plot_roc



# import or paste here your function cv_svm
# TODO
def cv_svm():
    pass


def explained_var(xtrain: np.ndarray) -> np.ndarray:
    """Compute and plot the cumulative explained variance ratio for PCA.

    Calculate and plot the cumulative explained variance ratio for
    PCA applied to the training data.


    Args:
        xtrain (np.ndarray): The training data.

    Returns:
        np.ndarray: An array containing the cumulative explained variance ratios.
    """
    # 4. initialize PCA and fit on train data
    # TODO

    # 5. plot cumulative explained variance ratios of each principal component
    # against the number of components (given: 'plot_expl_var')
    # TODO
    # 6. return array of cumulative explained variance ratios
    return None  # TODO


def pca_train(
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    n_comp: int,
    train_fun: Callable[[np.ndarray, np.ndarray], Any],
) -> tuple[PCA, Any]:
    """Train a model with a given train function on the PCA-transformed data.

    Extract the top 'n_comp' principal components from the training data using PCA,
    and then trains a model on these components as features.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.
        n_comp (int): The number of PCA components.
        train_fun (Callable): The function used to train the model;
            it takes the training data and labels and returns the trained model.

    Returns:
        tuple: A tuple containing the PCA decomposition
               with 'n_comp' components and the model trained with the 'train_fun' function.
    """
    # 8. initialize PCA and fit on train data
    # TODO

    # 9. transform input data using PCA transform
    # TODO
    # 10. train model on transformed PCA features
    # TODO
    # 11. return PCA decomposition object and trained model
    return None  # TODO


# (optional)
def select_n_comp(xtrain: np.ndarray, ytrain: np.ndarray, comps: np.ndarray) -> int:
    """Select the best number of PCA components with a cross-validated grid search.

    For every candidate in 'comps' the mean accuracy of a fixed SVM classifier
    is estimated with a 5-fold cross-validation. Scaling and PCA are fitted
    inside each fold on the training part only. In case of a tie, the smallest
    number of components wins.

    Note: this function only selects, it does not return a score. Its mean
    accuracies are optimistically biased, because the same data is used for
    selecting and for scoring. Use 'nested_cv' to estimate the performance.

    Args:
        xtrain (np.ndarray): The unscaled training data.
        ytrain (np.ndarray): The training labels.
        comps (np.ndarray): Array of numbers of PCA components to consider.

    Returns:
        int: The best number of PCA components.
    """
    # 20. define inner 5-fold cross-validation strategy
    # (stratified and shuffled, since the classes are imbalanced and may be sorted)
    # TODO

    # 21. initialize the array collecting the summed accuracy per candidate
    # TODO
    # 22. loop over the folds of the inner cross-validation
    # TODO
    # 22.1. generate current train and validation set out of the given data
    # TODO
    # 22.2. fit 'StandardScaler' on the training part
    # and scale both the training and the validation part
    # TODO

    # 22.3. fit the PCA once with the largest candidate;
    # since the components are sorted and whitened individually,
    # using the first n_comp columns is equivalent to fitting PCA(n_comp)
    # TODO
    # 22.4. train a classifier per candidate and accumulate its accuracy
    # TODO

    # 23. return the candidate with the highest mean accuracy
    return None  # TODO

# (optional)
def nested_cv(
    x: np.ndarray, y: np.ndarray, comps: np.ndarray
) -> tuple[float, float, list[int]]:
    """Estimate the performance of the whole pipeline with a nested cross-validation.

    The outer cross-validation evaluates, the inner one (inside 'select_n_comp')
    selects the number of components. Since the number of components is chosen
    on the outer training part only, the returned accuracy is an unbiased
    estimate of the procedure 'scale, choose n_comp by CV, train an SVM' --
    not of one particular model.

    Args:
        x (np.ndarray): The unscaled data.
        y (np.ndarray): The labels.
        comps (np.ndarray): Array of numbers of PCA components to consider.

    Returns:
        tuple: The mean and the standard deviation of the outer accuracies and
               the list of the numbers of components chosen in each outer fold.
    """
    # 24. define the outer 5-fold cross-validation strategy
    # TODO
    # 25. loop over the outer folds
    # TODO
    # 25.1. generate current train and test set out of the given data
    # TODO

    # 25.2. select the number of components on the outer training set only
    # TODO

    # 25.3. scale, fit the PCA and train the classifier on the outer training set
    # TODO

    # 25.4. evaluate on the outer test fold
    # TODO

    # 26. return mean, standard deviation and the components chosen per fold
    return None # TODO



if __name__ == "__main__":
    # 1. load dataset 'Labeled Faces in the Wild' and get data,
    # take only classes with at least 70 images; downsize images for speed up
    # TODO

    # 2. split data into training and test data
    # TODO

    # 3. use 'StandardScaler' on train data and scale both train and test data
    # (keep the unscaled training data for the optional part)
    # TODO

    # 7. use `explained_var` function to calculate minimum number of components
    # needed to capture 90% of the variance; print this number
    # TODO

    # 12. import or paste your cv_svm function above,
    # use it together with computed number of components
    # to call 'pca_train' and train model with reduced number of features,
    # measure time duration
    # TODO

    # 13. transform input test data using PCA transform
    # TODO
    # 14. compute and print accuracy of best model on test set
    # TODO

    # 15. use function 'cv_svm' to perform hyperparameter search with cross validation
    # on non-preprocessed train set, measure time
    # TODO

    # 16. compute and print accuracy of best model on test set
    # TODO

    # 17. (optional) plot top 12 eigenfaces
    # TODO

    # 18. (optional) plot ROC curves
    # TODO

    # (optional)
    # 27. generate list of candidate numbers of components for the grid search
    # TODO

    # 28. estimate the performance of the whole procedure with 'nested_cv'
    # on the (unscaled) training data
    # TODO

    # 29. select the number of components for the final model on the whole
    # (unscaled) training set -- the test set must not be used for model selection
    # TODO

    # 30. repeat steps 12 to 14 with this number
    # TODO

    # 31. (optional) the same nested cross-validation with a scikit-learn pipeline
    # TODO
    
    pass
