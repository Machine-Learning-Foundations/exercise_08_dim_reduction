"""Use PCA to improve soft-margin SVM for face recognition."""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# import or paste here your function cv_svm
# TODO


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

    # 5. plot cumulative explained variance ratios
    # of each principal component against the number of components
    # TODO
    # 6. return array of cumulative explained variance ratios
    return None  # TODO


def pca_train(
    xtrain: np.ndarray, ytrain: np.ndarray, n_comp: int, train_fun: GridSearchCV
) -> tuple[PCA, GridSearchCV]:
    """Train a model with a given train function on the PCA-transformed data.

    Extract the top 'n_comp' principal components from the training data using PCA,
    and then trains a model on these components as features.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.
        n_comp (int): The number of PCA components.
        train_fun: The estimator function used to train the model.

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
def gs_pca(x: np.ndarray, y: np.ndarray, comps: np.ndarray) -> int:
    """Perform grid search to find the best number of PCA components.

     Conduct a grid search to identify the optimal number of PCA components
     to use for feature dimensionality reduction before the training.

    Args:
        x (np.ndarray): The input data.
        y (np.ndarray): The target labels.
        comps (np.ndarray): Array of numbers of PCA components to consider.

    Returns:
        int: The best number of PCA components.
    """
    # 20. define outer 5-fold cross-validation strategy
    # TODO

    # 21. initialize variables to track best score and components
    # TODO
    # 22. loop over specified number of PCA components to consider
    # TODO
    # 22.1. create outer cross-validation loop iterating over splits
    # TODO
    # 22.1.1 take indices and generate current train and test set out of given data
    # TODO

    # 22.1.2. fit 'StandardScaler' on train data
    # and scale both train and test data
    # TODO
    # 22.1.3. create `PCA` instance with same parameters as before
    # and transform train data
    # TODO

    # 22.1.4. here either call 'cv_cvm' function
    # or initialize SVM classifier directly with parameters
    # TODO
    # 22.1.5. predict labels on test data and compute accuracy score for each fold
    # TODO

    # 22.2. calculate mean accuracy score  across all folds
    # for current number of components
    # TODO
    # 22.3. check if this is best number of components based
    # on mean accuracy and update if needed
    # TODO
    # 23. return best number of PCA components found
    return None  # TODO


if __name__ == "__main__":
    # 1. load dataset 'Labeled Faces in the Wild' and get data,
    # take only classes with at least 70 images; downsize images for speed up
    # TODO

    # 2. split data into training and test data
    # TODO

    # 3. use 'StandardScaler' on train data and scale both train and test data
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
    # 24. generate list of parameters to perform grid search
    # TODO
    # 25. call 'gs_pca' to determine best number of components and print it
    # TODO

    # 26. repeat steps 12 to 14
    # TODO
    pass
