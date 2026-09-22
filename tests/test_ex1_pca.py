"""Test pca functions."""

import sys

import numpy as np
import pytest
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA

sys.path.insert(0, "./src/")
from src.ex1_pca import expl_var, pca_inverse_transform, pca_transform

# sample image for testing, reshaped to (d, n) = (num_rows, num_columns * num_channels)
image = load_sample_image("flower.jpg")
image_rows = np.reshape(image, (image.shape[0], -1))
d, n = image_rows.shape


@pytest.fixture(scope="module")
def custom_pca():
    """Run the custom PCA once for all tests."""
    return pca_transform(image_rows)


@pytest.fixture(scope="module")
def sklearn_pca():
    """Fit sklearn's PCA on the same data (sklearn expects samples as rows)."""
    return PCA(svd_solver="full").fit(image_rows.T)


def test_pca_transform_shapes(custom_pca):
    """Test the shapes of all returned arrays."""
    eigenvalues, eigenvectors, centered_data, mean_vector = custom_pca
    assert eigenvalues.shape == (d,)
    assert eigenvectors.shape == (d, d)
    assert centered_data.shape == (d, n)
    assert mean_vector.shape == (d, 1)


def test_pca_transform_mean_and_centering(custom_pca):
    """Test the feature means and that the data is centered."""
    _, _, centered_data, mean_vector = custom_pca
    np.testing.assert_allclose(mean_vector[:, 0], image_rows.mean(axis=1))
    np.testing.assert_allclose(centered_data.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(centered_data + mean_vector, image_rows, atol=1e-10)


def test_pca_transform_eigenvalues(custom_pca, sklearn_pca):
    """Test that the eigenvalues are sorted and equal to sklearn's."""
    eigenvalues = custom_pca[0]
    assert np.all(np.diff(eigenvalues) <= 0), "Eigenvalues are not sorted descending."
    np.testing.assert_allclose(
        eigenvalues,
        sklearn_pca.explained_variance_,
        rtol=1e-3,  # also accepts the biased estimate (ddof=0)
        atol=1e-9 * sklearn_pca.explained_variance_[0],
    )


def test_pca_transform_eigenvectors(custom_pca, sklearn_pca):
    """Test the eigenvectors.

    Eigenvectors are only unique up to sign, so we check orthonormality,
    the eigenvector equation and the agreement with sklearn up to sign.
    """
    eigenvalues, eigenvectors, centered_data, _ = custom_pca
    # orthonormal columns
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(d), atol=1e-8)
    # the eigenvectors (columns) diagonalize the covariance matrix: V^T C V = diag
    projected_cov = eigenvectors.T @ np.cov(centered_data) @ eigenvectors
    off_diagonal = projected_cov - np.diag(np.diag(projected_cov))
    assert np.max(np.abs(off_diagonal)) < 1e-8 * eigenvalues[0]
    np.testing.assert_allclose(
        np.diag(projected_cov), eigenvalues, rtol=1e-3, atol=1e-9 * eigenvalues[0]
    )
    # leading components agree with sklearn up to sign
    k = 10
    dots = np.abs(np.sum(eigenvectors[:, :k] * sklearn_pca.components_[:k].T, axis=0))
    np.testing.assert_allclose(dots, 1, atol=1e-6)


@pytest.mark.parametrize("n_components", [0, 1, 10, 50])
def test_pca_inverse_transform(custom_pca, n_components):
    """Test the inverse PCA transformation against sklearn.

    Unlike the eigenvectors, the reconstruction is unique (independent of signs),
    so it can be compared with a tight tolerance.
    """
    _, eigenvectors, centered_data, mean_vector = custom_pca
    custom_reconstructed_data = pca_inverse_transform(
        centered_data, eigenvectors, mean_vector, n_components
    )
    assert custom_reconstructed_data.shape == image_rows.shape

    if n_components == 0:
        # without any component only the mean remains
        expected = np.broadcast_to(mean_vector, image_rows.shape)
    else:
        sk_pca = PCA(svd_solver="full", n_components=n_components).fit(image_rows.T)
        expected = sk_pca.inverse_transform(sk_pca.transform(image_rows.T)).T
    np.testing.assert_allclose(custom_reconstructed_data, expected, atol=1e-6)


def test_pca_inverse_transform_all_components(custom_pca):
    """Test that all components reconstruct the original data."""
    _, eigenvectors, centered_data, mean_vector = custom_pca
    reconstructed = pca_inverse_transform(centered_data, eigenvectors, mean_vector, d)
    np.testing.assert_allclose(reconstructed, image_rows, atol=1e-6)


@pytest.mark.parametrize(
    ("n_components", "expected"),
    [(0, 0.0), (1, 0.5 / 1.1), (2, 0.8 / 1.1), (4, 1.0)],
)
def test_expl_var(n_components, expected):
    """Test the computation of the explained variance ratio."""
    eigenvalues = np.array([0.5, 0.3, 0.2, 0.1])
    cumulative_explained_var = expl_var(eigenvalues, n_components)
    assert np.isclose(cumulative_explained_var, expected)
