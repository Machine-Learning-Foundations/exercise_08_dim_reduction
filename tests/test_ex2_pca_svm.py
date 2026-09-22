"""Test pca svm methods."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, "./src/")
from src.ex2_pca_svm import cv_svm, explained_var, nested_cv, pca_train, select_n_comp


# auxiliary function for testing the pca_train function
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
    """Test the cross-validated soft margin SVM classifier.

    Note: the expected values assume the parameter grid from Day 06
    (C in 10^-2, ..., 10^3 and kernel in {'rbf', 'linear', 'poly'}).
    """
    x_t, y_t = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    clf = cv_svm(x_t, y_t)

    # check if returned object is a fitted grid search
    assert isinstance(clf, GridSearchCV)
    assert hasattr(clf, "best_params_")
    assert hasattr(clf, "predict")

    # check best results
    assert np.isclose(clf.best_score_, 0.99)
    assert clf.best_params_["C"] == 1


def test_explained_var():
    """Test the cumulative explained variance array."""
    np.random.seed(0)
    num_samples, num_features = 100, 10
    synthetic_data = np.random.rand(num_samples, num_features)

    cumulative_explained_var = explained_var(synthetic_data)
    plt.close("all")

    assert isinstance(cumulative_explained_var, np.ndarray)
    assert cumulative_explained_var.shape == (num_features,)
    # cumulative values are non-decreasing and end at 1
    assert np.all(np.diff(cumulative_explained_var) >= -1e-12)
    assert np.isclose(cumulative_explained_var[-1], 1.0)
    # check first element
    assert np.isclose(cumulative_explained_var[0], 0.16132983)


def test_pca_train():
    """Test training a model on PCA-transformed features."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=29
    )
    n_components = 2

    pca, model_pca = pca_train(x_train, y_train, n_components, train_knn)

    # check the fitted PCA
    assert hasattr(pca, "components_")
    assert pca.n_components_ == n_components
    assert pca.whiten
    # check that train_fun was used to train the model
    assert isinstance(model_pca, KNeighborsClassifier)
    assert model_pca.n_features_in_ == n_components

    # test model on test set
    y_pred = model_pca.predict(pca.transform(x_test))
    accuracy = np.mean(y_pred == y_test)
    assert np.isclose(accuracy, 0.9210526)


def test_select_n_comp():
    """Test the cross-validated selection of the number of principal components.

    Note: the expected value assumes
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    and a fixed SVC(C=10, kernel="rbf").
    """
    numbers = np.arange(1, 5, 1)
    best_num_comp = select_n_comp(x, y, numbers)
    assert best_num_comp in numbers
    assert best_num_comp == 4


def test_nested_cv():
    """Test the nested cross-validation."""
    numbers = np.arange(1, 5, 1)
    mean_acc, std_acc, chosen_comps = nested_cv(x, y, numbers)

    # one selected number of components per outer fold
    assert len(chosen_comps) == 5
    assert all(n in numbers for n in chosen_comps)
    # the estimate itself
    assert np.isclose(mean_acc, 0.94)
    assert np.isclose(std_acc, 0.04422166, atol=1e-6)


def test_nested_cv_uses_outer_training_data_only(monkeypatch):
    """Test that the components are selected without the outer test fold.

    'select_n_comp' must be called on the outer training part only (4/5 of the
    data), otherwise the selection leaks information from the outer test fold.
    """
    import src.ex2_pca_svm as ex2

    sizes = []
    original = ex2.select_n_comp

    def spy(xtrain, ytrain, comps):
        sizes.append(len(xtrain))
        return original(xtrain, ytrain, comps)

    monkeypatch.setattr(ex2, "select_n_comp", spy)
    ex2.nested_cv(x, y, np.arange(1, 5, 1))

    assert len(sizes) == 5, "select_n_comp should be called once per outer fold."
    assert all(size == 120 for size in sizes), (
        "select_n_comp must only see the outer training part (120 of 150 samples)."
    )
