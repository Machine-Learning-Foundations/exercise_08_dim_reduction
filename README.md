# Dimensionality Reduction Exercise

In this exercise, we will take a closer look at the mechanics of Principal Component Analysis (PCA). We will explore how PCA can reduce the complexity of our data and understand the practical benefits of this dimensionality reduction technique. Our first goal is to project our high-dimensional data onto a more compact feature space. We will then visualize how, even with this reduced set of features, we can retain the most information. This insight will serve as a basis for preprocessing the data that was used in our previous Support Vector Classification (SVC) exercise. We will observe the impact of this dimensionality reduction on our subsequent SVC training.

### Task 1: Principal Component Analysis

In this task, we will implement all the necessary steps to perform a PCA and visualize how much of the original information content of an image remains after the image features are projected into a lower dimensional space. 

**Provided helper functions** (in `src/util_pca.py`, already imported in `src/ex1_pca.py`), so you do not have to write any plotting or file-export code in this task:

- `to_uint8(image)` returns the image as `uint8`. The reconstructed images are float arrays whose values are not guaranteed to lie in $[0, 255]$, especially for small _k_. Casting them directly would make them wrap around (e.g. $-3$ would become $253$), which is why this function rounds and clips them to the valid range first.
- `save_image(image, path)` converts the image with `to_uint8` and writes it with `imsave` from `skimage.io`.
- `plot_expl_var_ssim(n_components, expl_vars, ssims, path=None)` plots the cumulative explained variance ratio and the SSIM against the number of components, the SSIM on a second y-axis, and saves the figure if a path is given. It expects three sequences of the same length.

Navigate to `src/ex1_pca.py` and have a look at the `__main__` function :

1. Create an empty directory called ``output`` (Hint: `os.makedirs`).
2. Load the `statue.jpg` image from `data/images/` using ``imageio.imread`` and plot it with ``plt.imshow()``. Use the function ``save_image`` from `src/util_pca.py` to save the image into the ``output`` directory as ``original.png`` with ``save_image(image, "./output/original.png")``.

> Note: If you plot inside a loop, remember to ``plt.close()`` afterwards: otherwise every ``imshow`` stacks another image into the same figure, so the plotting gets slower and slower and all of the images stay in memory.

3. Reshape the image array into a 2D-array of shape $(d,n)$, where $d$ = `num_rows` is the number of features and $n$ = `num_columns * num_channels` would represent our examples.

Now we will implement the functions to perform a PCA transform and an inverse transform on our 2D array. First implement the function `pca_transform`: 

4. Compute the mean vector over the features of the input matrix. The resulting mean vector should have the size $(d,1)$. (Hint: use `keepdims=True`in the function `numpy.mean` to keep the dimensions for easier subtraction.)
5. Center the data by subtracting the mean from the 2D image array.
6. Compute the covariance matrix of the centered data. (Hint: `numpy.cov`.)
7. Perform the eigendecomposition of the covariance matrix. (Hint: `numpy.linalg.eigh`)
8. Sort eigenvalues in descending order and eigenvectors by their descending eigenvalues.
9. Return sorted eigenvalues, eigenvectors, centered data and the mean vector.

Next, implement the function `pca_inverse_transform`, which reconstructs the data using the top $n_{comp}$ principal components following these steps: 

10. Select the first $n_{comp}$ components from the given eigenvectors. 
11. Project the centered data onto the space defined by the selected eigenvectors by multiplying the transposed selected eigenvectors and the centered data matrix, giving us the reduced data.
12. Reconstruct the data projecting it back to the original space by multiplying the selected eigenvectors with the reduced data. Don't forget to add the mean vector afterwards.
13. Return the reconstructed data.

Now, before returning to the `__main__` function, we also want to calculate the explained variance associated with our principal components. For that, implement the `expl_var` function following these steps:

14. Calculate the total variance by summing up all the eigenvalues.
15. Compute the cumulative explained variance by summing the first $n_{comp}$ eigenvalues.
16. Determine the cumulative explained variance ratio by dividing the cumulative explained variance by the total variance. Return the result.

Go back to the `__main__` function and implement the following TODOs:

17. Loop through a range of all possible values of the number of components, i.e. from $0$ up to and including $d$. It is sufficient to use the step size of 10 to speed up the process (but make sure to include $d$ itself). To monitor the progress of the loop, you can create a progress bar using [the very handy Python package tqdm](https://github.com/tqdm/tqdm).

	17.1. Perform PCA using the previously implemented `pca_transform` function. Since the eigendecomposition does not depend on the number of components, do this only once *before* the loop.

	17.2. Apply  the `pca_inverse_transform` function to project the image to lower-dimensional space using the current number of components and reconstruct the image from this reduced representation.

	17.3. Bring the resulting array back into the original image shape. Use ``save_image`` to convert it with ``to_uint8`` and save it in the ``output`` folder with ``save_image(recovered_image, f"./output/pca_{n_components}.png")``, so that the file is called ``pca_k.png``, where _k_ is replaced with the number of components used to create the image.

	> Note: Keep the converted `uint8` image, you will compare it with the original in step 17.5. The SSIM should be computed on the same data that you saved, not on the raw float array.
   
	17.4. Compute the cumulative explained variance ratio for the current number of components using the `expl_var` function and store it in a list for later plotting.

	17.5. We would also like to quantify how closely our created image resembles the original one. Use ``skimage.metrics.structural_similarity`` to compute a perceptual similarity score (SSIM) between the original and the reconstructed image and also store it in another list for later plotting. As we deal with RGB images, you have to pass `channel_axis=2` to the SSIM function.
   
18./19. Plot both curves against the number of components by calling ``plot_expl_var_ssim(n_components_list, cumulative_explained_vars, ssims, "./output/explained_variance_ssim.png")``. Have a look at the implementation in `src/util_pca.py` if you are curious how the second y-axis is done; the [matplotlib gallery](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html) has an example as well.
20. Look through the images you generated and find the one with the smallest _k_ which you would deem indistinguishable from the original. Compare this to both the explained variance and SSIM curves.
21. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.


### Task 2: PCA as Pre-processing

We have seen that a significantly reduced feature dimensionality is often sufficient to effectively represent our data, especially in the case of image data. Building upon this insight, we will now revisit our Support Vector Classification (SVC) task from Day 05, but this time with a preprocessing of our data using a PCA. Again, we will use the [Labeled Faces in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/).

**Provided helper functions** (in `src/util_pca.py`): `plot_expl_var(cumulative_explained_var)` plots the cumulative explained variance ratios against the number of components; `plot_image_matrix` and `plot_roc` are used in the optional steps 17 and 18. As in Task 1, you do not have to write any plotting code.

We start in the the `__main__` function.

1. Load the dataset from ``sklearn.datasets.fetch_lfw_people`` in the same way as for Task 2 of Day 05 and get access to the data.
2. Split the data 80:20 into training and test data. Use `random_state=42` in the split function. 
3. Use the `StandardScaler` from `sklearn.preprocessing` on the train set and scale both the train and the test set. Keep a copy of the unscaled training data, you will need it in the optional part.

Our goal now is to determine the minimum number of principal components needed to capture at least 90% of the variance in the data. First, implement the `explained_var` function:

4. Create an ``sklearn.decomposition.PCA`` instance and fit it to the data samples. Set `random_state=42` and `whiten=True` to normalize the components to have unit variance.
5. Sum up the ``explained_variance_ratio_`` property of the ``PCA`` instance to get cumulative values (e.g. using ``np.cumsum``). Plot them against the number of components by calling ``plot_expl_var(cumulative_explained_var)``.
6. Return the array of cumulative explained variance ratios.

7. Return to the `__main__` function and use the `explained_var` function to calculate the minimum number of components needed to capture 90% of the variance. Print this number.
	
Implement the `pca_train` function to train a model on preprocessed data: 

8. Create a ``PCA`` instance and fit it to the data samples extracting the given number of components. Set `random_state=42` and `whiten=True`.
9. Project the input data on the orthonormal basis using the `PCA.transform`, resulting in a new dataset, where each sample is represented by the given number of the top principal components.
10. Call the `train_fun` function, which is passed as an argument, to train a model on the transformed PCA features.
11. The function should return a tuple containing two elements: the PCA decomposition object and the trained model.

12. Import or paste your cv_svm function from Task 2 of Day 06 above the code. Utilize it together with the computed number of required components to call `pca_train` in the `__main__` function.  This will allow us to train the model with the reduced feature set. Use the `time` function from the `time` module to measure and print the duration of this process for evaluation. 
	
13. To evaluate the model on the test set, we need to perform the same transform on the test data, as we did on the training data. Use the `PCA.transform` of your PCA decomposition object to do this.
14. Now we can compute and print the accuracy of our trained model on the test set.
	
15. In order to compare this model with the one without the PCA preprocessing, apply the function `cv_svm` on the original training set and measure the time.
	
16. Compute and print the accuracy of this trained model on the test set.
	
17. (Optional) You can use the `plot_image_matrix` function from `src/util_pca.py` to plot the top 12 eigenfaces.
	
18. (Optional) Furthermore, you can use the `plot_roc` function from `src/util_pca.py` to plot the ROC curves of both models.
	
19. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.


#### (Optional) Nested Cross-Validation for the best Number of Principal Components

We have seen how PCA can improve both the runtime and the results of our training. So far we fixed the number of principal components to the value from step 7. You can practice your coding skills by tuning this number with a cross-validation as well.

Two things have to be kept apart here, and this is the point of this part:

- **Selecting** the number of components. An inner cross-validation compares the candidate values on the training data. This is what the `select_n_comp` function does below.
- **Evaluating** the resulting procedure. Whoever selects on a data split cannot use the same split to report the performance: the winner of a comparison is optimistically biased. Therefore an outer cross-validation wraps around the selection and only ever evaluates on data that the selection has not seen. This is the `nested_cv` function, and this nesting of the two loops is what makes a cross-validation a *nested* one.

The test set from step 2 stays untouched in both functions. It is reserved for the final evaluation in step 30; using it for model selection would lead to an overly optimistic accuracy (data leakage).

To save runtime we fix the SVM hyperparameters in the innermost training step (`C=10` and `kernel='rbf'`), which we determined for you beforehand. If you replace this by a call to `cv_svm`, you get an additional cross-validation level that also tunes `C` and the kernel.

Implement the `select_n_comp` function:

20. Define the inner k-fold cross-validation strategy with 5 folds using `StratifiedKFold` from `sklearn.model_selection`. Set `shuffle=True` and `random_state=42`. Stratification keeps the class proportions in each fold, which matters because the classes in our dataset are imbalanced.
21. Initialize an array in which you accumulate the accuracy of every candidate number of components over the folds.
22. Iterate through the splits of the inner cross-validation, obtaining the training and validation indices for each split.

	22.1. Generate the current training and validation sets from the given data based on these indices.

	22.2. Scale the generated data fitting the `StandardScaler` on the training part and scale both parts. Note that the scaler has to be fitted inside the loop: fitting it on all the data beforehand would leak information from the validation part into the training.

	22.3. Instantiate a `PCA` object (again with `random_state=42` and `whiten=True`) and fit it on the training part. You only need a single PCA per fold, fitted with the *largest* candidate number of components: because the components are sorted by their eigenvalue and `whiten` scales each component individually, using the first $k$ columns of the transformed data is equivalent to fitting a `PCA` with $k$ components. This makes the grid over the candidates almost free.

	22.4. For each candidate $k$, train an `sklearn.svm.SVC` with `C=10` and `kernel='rbf'` on the first $k$ columns of the transformed training part, predict the labels of the validation part and add the accuracy to your array.

23. Return the candidate with the highest summed (equivalently: mean) accuracy. Note that `np.argmax` returns the first maximum, so in case of a tie the smallest number of components wins.

Now implement the `nested_cv` function:

24. Define the outer 5-fold cross-validation strategy, again with `StratifiedKFold`, `shuffle=True` and `random_state=42`.
25. Iterate through the splits of the outer cross-validation.

	25.1. Generate the current training and test sets from the given data based on the indices.

	25.2. Call `select_n_comp` on the outer **training** set only and store the returned number of components.

	25.3. Scale the data (fitting the `StandardScaler` on the outer training set), fit a `PCA` with the selected number of components and train an `SVC` with `C=10` and `kernel='rbf'`.

	25.4. Transform the outer test set, predict its labels and store the accuracy of this fold.

26. Return the mean and the standard deviation of the outer accuracies together with the list of the numbers of components chosen in each fold.

Go back to the `__main__` function.

27. Generate the list of the candidate numbers of components, consisting of the following numbers: `[c-10 c-5 c c+5 c+10]`, where $c$ is the number determined in step 7.
28. Use `nested_cv` on the **unscaled** training data (the scaling happens inside the folds) and print the mean accuracy, its standard deviation and the numbers of components chosen per fold. The mean is your honest estimate for the procedure as a whole, not for a single model. Have a look at the numbers chosen per fold: if they differ a lot, this tells you that the accuracy is rather insensitive to the exact number of components, and you should not over-interpret the single winner from the next step.
29. Now determine the number of components for the final model by calling `select_n_comp` on the whole (unscaled) training set, and print it.
30. Repeat the steps 12 to 14 with this number and compare the new accuracy.

31. (Optional) The same nested cross-validation can be expressed with a `sklearn.pipeline.Pipeline` in a few lines: put `StandardScaler`, `PCA` and `SVC` into a pipeline, wrap it into a `GridSearchCV` over the parameter `pca__n_components` (this is the inner cross-validation), and pass the whole grid search to `sklearn.model_selection.cross_val_score` with the outer cross-validation. Compare the result to your own implementation. Note that the pipeline rules out the leakage of step 22.2 by construction, because every step is refitted on the training part of each split.
32. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.
