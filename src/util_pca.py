"""Utility functions for pca."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.metrics import auc, roc_curve


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
        plt.imshow(images[indices[i]].reshape((h, w)), cmap=cm.get_cmap("gray"))
        plt.title(titles[indices[i]], size=12)
        plt.xticks(())
        plt.yticks(())


def plot_roc(model, x_test, y_test, n_classes, target_names):
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
