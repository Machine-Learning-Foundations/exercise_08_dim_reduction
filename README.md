# Dimensionality Reduction Exercise

In this exercise, we will take a closer look at the mechanics of Principal Component Analysis (PCA). We will explore how PCA can reduce the complexity of our data and understand the practical benefits of this dimensionality reduction technique. Our first goal is to project our high-dimensional data onto a more compact feature space. We will then visualize how, even with this reduced set of features, we can retain the most information. This insight will serve as a basis for preprocessing the data that was used in our previous Support Vector Classification (SVC) exercise. We will observe the impact of this dimensionality reduction on our subsequent SVC training.

### Task 1: Principal Component Analysis

In this task, we will implement all the necessary steps to perform a PCA and visualize how much of the original information content of an image remains after the image features are projected into a lower dimensional space. 

Navigate to `src/ex1_pca.py` and have a look at the `__main__` function :

1. Create an empty directory called ``output`` (Hint: `os.makedirs`).
2. Load the `statue.jpg` image from `data/images/` using ``imageio.imread``, plot it and save it into the ``output`` directory as ``original.png`` using `imsave` from `skimage.io`. (Hint: For `skimage.io` to be able to correctly interpret your image, you should cast it to the uint8 dtype, for example by using ``your_array.astype(np.uint8)``.)

> Note: You can also use ``plt.imshow()`` followed by ``plt.savefig(your_path)`` as a simple way to save the image. Do not forget to ``plt.close()`` your plot afterwards, because we will export a fair amount of images in this exercise.

3. Reshape the image array into a 2D-array of shape $(d,n)$, where $d$ = `num_rows` is the number of features and $n$ = `num_columns * num_channels` would represent our examples.

Now we will implement the functions to perform a PCA transform and an inverse transform on our 2D array. First implement the function `pca_transform`: 

4. Compute the mean vector over the features of the input matrix. The resulting mean vector should have the size $(d,1)$. (Hint: use `keepdims=True`in the function `numpy.mean` to keep the dimensions for easier subtraction.)
5. Center the data by subtracting the mean from the 2D image array.
6. Compute the covariance matrix of the centered data. (Hint: `numpy.cov`.)
7. Perform the eigendecomposition of the covariance matrix. (Hint: `numpy.linalg.eigh`)
8. Sort eigenvalues in descending order and eigenvectors by their descending eigenvalues.
9. Return sorted eigenvalues, eigenvectors, centered data and the mean vector.

Next, implement the function `pca_inverse_transform`, which reconstructs the data using the top $n_comp$ principal components following these steps: 

10. Select the first $n_comp$ components from the given eigenvectors. 
11. Project the centered data onto the space defined by the selected eigenvectors by multiplying the transposed selected eigenvectors and the centered data matrix, giving us the reduced data.
12. Reconstruct the data projecting it back to the original space by multiplying the selected eigenvectors with the reduced data. Don't forget to add the mean vector afterwards.
13. Return the reconstructed data.

Now, before returning to the `__main__` function, we also want to calculate the explained variance associated with our principal components. For that, implement the `expl_var` function following these steps:

14. Calculate the total variance by summing up all the eigenvalues.
15. Compute the cumulative explained variance by summing the first 'n_comp' eigenvalues.
16. Determine the cumulative explained variance ratio by dividing the cumulative explained variance by the total variance. Return the result.

Go back to the `__main__` function and implement the following TODOs:

17. Loop through a range of all possible values of the number of components. It is sufficient to use the step size of 10 to speed up the process. To monitor the progress of the loop, you can create a progress bar using [the very handy Python package tqdm](https://github.com/tqdm/tqdm).

	17.1. Perform PCA using the previously implemented `pca_transform` function.

	17.2. Apply  the `pca_inverse_transform` function to project the image to lower-dimensional space using the current number of components and reconstruct the image from this reduced representation.

	17.3. Bring the resulting array back into the original image shape and save it in the ``output`` folder as an image called ``pca_k.png``, where _k_ is replaced with the number of components used to create the image.

	> Note: You should again cast the image back to the uint8 dtype.
   
	17.4. Compute the cumulative explained variance ratio for the current number of components using the `expl_var` function and store it in a list for later plotting.

	17.5. We would also like to quantify how closely our created image resembles the original one. Use ``skimage.metrics.structural_similarity`` to compute a perceptual similarity score (SSIM) between the original and the reconstructed image and also store it in another list for later plotting. As we deal with RGB images, you have to pass `channel_axis=2` to the SSIM function.
   
18. Plot the cumulative explained variances of each principal component against the number of components. 
19. Plot the SSIM values against the number of components. If you like a small matplotlib challenge, you can also try to add this curve with a second scale to the first plot (you can find an example on how to do this [in the matplotlib gallery](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html)).
20. Look through the images you generated and find the one with the smallest _k_ which you would deem indistinguishable from the original. Compare this to both the explained variance and SSIM curves.
21. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.

### Task 2: PCA as Pre-processing

We have seen that a significantly reduced feature dimensionality is often sufficient to effectively represent our data, especially in the case of image data. Building upon this insight, we will now revisit our Support Vector Classification (SVC) task from Day 06, but this time with a preprocessing of our data using a PCA. Again, we will use the [Labeled Faces in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/).

We start in the the `__main__` function.

1. Load the dataset from ``sklearn.datasets.fetch_lfw_people`` in the same way as for Task 2 of Day 05 and get access to the data.
2. Split the data 80:20 into training and test data. Use `random_state=42` in the split function. 
3. Use the `StandardScaler` from `sklearn.preprocessing` on the train set and scale both the train and the test set.

Our goal now is to determine the minimum number of principal components needed to capture at least 90% of the variance in the data. First, implement the `explained_var` function:

4. Create an ``sklearn.decomposition.PCA`` instance and fit it to the data samples. Set `random_state=42` and `whiten=True` to normalize the components to have unit variance.
5. Plot the cumulative explained variance ratios of each principal component against the number of components using the ``explained_variance_ratio_`` property of the ``PCA`` instance. Note, that you have to sum up these ratios to get cumulative values (e.g. using ``np.cumsum``).
6. Return the array of cumulative explained variance ratios.

7. Return to the `__main__` function and use the `explained_var` function to calculate the minimum number of components needed to capture 90% of the variance. Print this number.
	
Implement the `pca_train` function to train a model on preprocessed data: 

8. Create a ``PCA`` instance and fit it to the data samples extracting the given number of components. Set `random_state=42` and `whiten=True`.
9. Project the input data on the orthonormal basis using the `PCA.transform`, resulting in a new dataset, where each sample is represented by the given number of the top principal components.
10. Call the `train_fun` function, which is passed as an argument, to train a model on the transformed PCA features.
11. The function should return a tuple containing two elements: the PCA decomposition object and the trained model.

12. Import or paste your cv_svm function from Task 2 of Day 05 above the code. Utilize it together with the computed number of required components to call `pca_train` in the `__main__` function.  This will allow us to train the model with the reduced feature set. Use the `time` function from the `time` module to measure and print the duration of this process for evaluation. 
	
13. To evaluate the model on the test set, we need to perform the same transform on the test data, as we did on the training data. Use the `PCA.transform` of your PCA decomposition object to do this.
14. Now we can compute and print the accuracy of our trained model on the test set.
	
15. In order to compare this model with the one without the PCA preprocessing, apply the function `cv_svm` on the original training set and measure the time.

16. Compute and print the accuracy of this trained model on the test set.

17. (Optional) You can use the `plot_image_matrix` function from `src/util_pca.py` to plot the top 12 eigenfaces.
	
18. (Optional) Furthermore, you can use the `plot_roc` function from `src/util_pca.py` to plot the ROC curves of both models.
	
19. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.


#### (Optional) Grid Search and Nested CV for best Number of Principal Components

We have seen how PCA can improve both the runtime and the results of our training. You can practice your coding skills by implementing nested cross-validation. While we already have the inner cross-validation for hyperparameter tuning, we can also employ an outer cross-validation to determine the optimal number of principal components to include in our analysis.

Implement the `gs_pca` function, which utilizes a grid search approach to determine the most suitable number of PCA components for feature dimensionality reduction before the training:

20. Define the outer k-fold cross-validation strategy with 5 folds using `KFold` from `sklearn.model_selection`.
21. Next, initialize the variables to keep track of the best mean accuracy score and the corresponding number of PCA components found 22. Iterate through the specified list of PCA component values.

	22.1. Create an outer 5-fold cross-validation loop, iterating through the 5 splits while obtaining the training and testing indices for each split.
	
		22.1.1. Generate the current training and testing sets from the given data based on these indices.
		
		22.1.2. Scale the generated data fitting the `StandardScaler` on the training set and scale the training and test sets.
		
		22.1.3. Instantiate a PCA object with the same parameters as before and transform the training data.
		
		22.1.4. Now is the time to call our function `cv_svm` and perform the inner cross-validation to tune hyperparameters. In order to save you the time, we have determined that the following parameters consistently yield the best results: C=10 and kernel='rbf'. Therefore, you can skip the inner cross-validation step and proceed to create and train your classifier with these predefined parameters.
		
		22.1.5. Predict the labels on the test data and compute the accuracy score for each fold. 
		
	22.2. Calculate the mean accuracy score across the folds.
	
	22.3. If the mean accuracy score for the current number of PCA components is higher than the best score seen so far, update the  best score and the best number components.
23. The function should return the number of PCA components that yielded the highest mean accuracy score during the grid search. This represents the optimal number of components for feature dimensionality reduction.

Go back to the `__main__` function. 

24. Generate the list of the parameters to perform the grid search, consisting of the following numbers: `[c-10 c-5 c c+5 c+10]`, where $c$ is the number determined in step 7.
25. Use the `gs_pca` function to determine the best number of components and print it.
26. Repeat the steps 12-14 with this best number and compare the new accuracy.	
27. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.


