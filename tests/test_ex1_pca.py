"""Test pca functions."""
import sys

import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA

sys.path.insert(0, "./src/")
from src.ex1_pca import expl_var, pca_inverse_transform, pca_transform

# sample image for testing
image = load_sample_image("flower.jpg")
image_rows = np.reshape(image, (image.shape[0], -1)).T


def test_pca_transform():
    """Test the PCA transformation of the image."""
    custom_eigenvalues, custom_eigenvectors, centered_data, mean_vector = pca_transform(
        image_rows
    )

    # use sklearn's PCA for same transformation
    sklearn_pca = PCA(svd_solver="full")
    sklearn_pca.fit(image_rows)
    sklearn_eigenvalues = sklearn_pca.explained_variance_

    # check if eigenvalues are approximately equal
    np.testing.assert_allclose(custom_eigenvalues, sklearn_eigenvalues, rtol=1)

    # check orthogonality of eigenvectors, since we cannot check if they are equal
    # due to not uniqueness of eigendcomposition
    for i in range(custom_eigenvectors.shape[1]):
        for j in range(i + 1, custom_eigenvectors.shape[1]):
            dot_product = np.dot(custom_eigenvectors[:, i], custom_eigenvectors[:, j])
            assert np.allclose(
                abs(dot_product), 0
            ), f"Eigenvectors {i} and {j} are not orthogonal."

    assert len(centered_data) == len(image_rows)
    assert len(mean_vector) == image_rows.shape[1]


def test_pca_inverse_transform():
    """Test the inverse PCA transformation."""
    n_components = 10
    _, custom_eigenvectors, centered_data, mean_vector = pca_transform(image_rows)
    custom_reconstructed_data = pca_inverse_transform(
        centered_data, custom_eigenvectors, mean_vector, n_components
    )

    # use sklearn's PCA for same transformation
    sklearn_pca = PCA(n_components=n_components)
    sklearn_pca.fit(image_rows)
    sklearn_reconstructed_data = sklearn_pca.inverse_transform(
        sklearn_pca.transform(image_rows)
    )

    assert custom_reconstructed_data.shape == image_rows.shape
    # check if results are approximately equal
    np.testing.assert_allclose(
        custom_reconstructed_data, sklearn_reconstructed_data, rtol=5
    )


def test_expl_var():
    """Test the computation of the explained variance ratio."""
    # generate some sample eigenvalues and set n_components
    eigenvalues = np.array([0.5, 0.3, 0.2, 0.1])
    n_components = 2

    cumulative_explained_var = expl_var(eigenvalues, n_components)

    # check if results are approximately equal
    assert np.allclose(cumulative_explained_var, 0.727272727)
