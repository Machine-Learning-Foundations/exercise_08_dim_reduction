"""Utility functions for pca."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from sklearn.metrics import auc, roc_curve


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image array to the uint8 dtype.

    The reconstructed images of Task 1 are float arrays whose values are not
    guaranteed to lie in [0, 255]. Casting them to uint8 directly would make
    them wrap around (e.g. -3 would become 253), so the values are rounded and
    clipped to the valid range first.

    Args:
        image (np.ndarray): The image with arbitrary float or integer values.

    Returns:
        np.ndarray: The image as uint8, with all values within [0, 255].
    """
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, path: str) -> None:
    """Save an image array as a .png file, converting it to uint8 beforehand.

    Args:
        image (np.ndarray): The image in its original shape (h x w x channels).
        path (str): The path of the file to write.
    """
    imsave(path, to_uint8(image))


def plot_expl_var_ssim(
    n_components: list[int] | np.ndarray,
    expl_vars: list[float],
    ssims: list[float],
    path: str | None = None,
) -> None:
    """Plot the cumulative explained variance ratio and the SSIM in one figure.

    Both curves are plotted against the number of components, the SSIM on a
    second y-axis (see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html).

    Args:
        n_components (list or np.ndarray): The numbers of components (x-axis).
        expl_vars (list): The cumulative explained variance ratio per number of components.
        ssims (list): The SSIM between original and reconstruction per number of components.
        path (str, optional): If given, the figure is saved to this path.
    """
    fig, ax1 = plt.subplots()
    ax1.plot(n_components, expl_vars, c="C0")
    ax1.set_xlabel("number of components")
    ax1.set_ylabel("explained variance", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(n_components, ssims, c="C1")
    ax2.set_ylabel("SSIM", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    fig.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close(fig)


def plot_expl_var(cumulative_explained_var: np.ndarray) -> None:
    """Plot the cumulative explained variance ratio against the number of components.

    Args:
        cumulative_explained_var (np.ndarray): The cumulative explained variance ratios.
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(cumulative_explained_var) + 1), cumulative_explained_var)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance ratio")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_image_matrix(images, titles, h, w, n_row=3, n_col=4) -> None:
    """Plot a matrix of images.

    Args:
        images (np.ndarray): The array of the images.
        titles (np.ndarray or list): The titles of the images.
        h (int): The height of one image.
        w (int): The width of one image.
        n_row (int): The number of rows of images to plot.
        n_col (int): The number of columns of images to plot.
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    indices = np.arange(n_row * n_col)
    # np.random.shuffle(indices)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[indices[i]].reshape((h, w)), cmap="gray")
        plt.title(titles[indices[i]], size=12)
        plt.xticks(())
        plt.yticks(())


def plot_roc(model, x_test, y_test, n_classes, target_names) -> None:
    """Plot the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC).

    Calculate and plot the ROC curve and AUC for a multiclass classification model.

    Args:
        model: The trained classification model.
        x_test (np.ndarray): The test data.
        y_test (np.ndarray): The true labels for the test data.
        n_classes (int): The number of classes in the classification problem.
        target_names (list): List of class labels.
    """
    y_score = model.decision_function(x_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = itertools.cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "gold",
            "mediumpurple",
            "indigo",
            "lime",
        ]
    )
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(
                target_names[i], roc_auc[i]
            ),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")

    plt.show()
