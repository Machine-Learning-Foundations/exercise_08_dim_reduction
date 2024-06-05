"""Test pca svm methods."""
import sys

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, "./src/")
from src.ex2_pca_svm import cv_svm, explained_var, gs_pca, pca_train


# auxilary function for testing gs_pca funciton
def train_knn(xtrain, ytrain):
    """Train k-NN with k = 3."""
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xtrain, ytrain)
    return knn


# load Iris dataset for testing
iris = load_iris()
x = iris.data
y = iris.target


def test_cv_svm():
    """Test the cross-validated soft margin SVM classifier."""
    # create dummy dataset
    x_t, y_t = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    # reshape data to fit into SVM
    x_t = x_t.reshape(-1, 20)
    # call function that is to be tested
    clf = cv_svm(x_t, y_t)

    # check if returned object is of expected type
    assert isinstance(clf, GridSearchCV)

    # get best score
    best_score = clf.best_score_
    # get best parameters
    best_param_c = clf.best_params_["C"]

    # perform assertions on best results
    assert clf is not None
    assert hasattr(clf, "best_params_")
    assert hasattr(clf, "predict")
    assert np.allclose(best_score, 0.99)
    assert np.allclose(best_param_c, 1)


def test_explained_var():
    """Test the explained variance array generation."""
    # create synthetic data
    num_samples = 100
    num_features = 10
    np.random.seed(0)
    synthetic_data = np.random.rand(num_samples, num_features)

    # calculate PCA explained variance
    cumulative_explained_var = explained_var(synthetic_data)

    # check if result is NumPy array
    assert isinstance(cumulative_explained_var, np.ndarray)

    # check if length of cumulative_explained_var matches number of features
    assert cumulative_explained_var.shape[0] == num_features

    # check if cumulative explained variance is between 0 and 1
    eps = np.finfo(cumulative_explained_var.dtype).resolution
    assert all(
        0 - eps <= explained_var <= 1 + eps
        for explained_var in cumulative_explained_var
    )

    # check if cumulative explained variance is non-decreasing
    assert all(
        cumulative_explained_var[i] <= cumulative_explained_var[i + 1]
        for i in range(num_features - 1)
    )

    # check if first element is approx. equal
    assert np.allclose(cumulative_explained_var[0], 0.16132983)


def test_pca_train():
    """Test the explained variance array generation."""
    # split dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=29
    )

    # choose the number of components for PCA
    n_components = 2

    # call pca_train function with KNN as the classifier
    pca, model_pca = pca_train(x_train, y_train, n_components, train_knn)

    # check if pca and model_pca are not None
    assert pca is not None
    assert model_pca is not None

    # Check if model_pca has necessary attributes
    assert hasattr(model_pca, "n_neighbors")

    # check if PCA has been fit
    assert hasattr(pca, "components_")

    # test model on test set
    x_test_pca = pca.transform(x_test)
    y_pred = model_pca.predict(x_test_pca)

    # calculate accuracy of the model
    accuracy = np.mean(y_pred == y_test)

    # check if accuracy is reasonable
    assert np.allclose(accuracy, 0.9210526)


def test_gs_pca():
    """Test the cross-validation on the number of principal components."""
    numbers = np.arange(1, 5, 1)
    best_num_comp = gs_pca(x, y, numbers)
    assert best_num_comp == 3
