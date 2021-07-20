# sklearn summary
## Supervised learning
### **KNN classification and regression**:
* `k`: number of neighbors.
	
example:
	
	cls = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)
	reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=4)
### Linear Models  
1. #### Ridge:
	* `alpha`: coefficients regularization (L2). default is 1. higher value means coefficients are closer to zero. when value is zero it's the same as normal plain linear regression.
		
		example:
		
	      ridge = sklearn.linear_model.Ridge(alpha=0.1)

2. #### Lasso:
	* `alpha`: coefficients regularization (L1). reduces certain coefficients to zero (could be considered as automatic feature selection). higher value means a simpler model. default value is 1

		example:
		
		  lasso = sklearn.linear_model.Lasso(alpha=0.01, max_iter=10_000)

3. #### LogisticRegression:
	* `C`: coefficients regularization value. higher value of C corresponds to lower regularization and higher complexity model. this means the higher the value of C, the higher the coefficients.
	* `penalty`: specify the norm of regularization. `'l1'` selects features, `'l2'` reduces all coefficients, `'elasticnet'` mix of 'L1' and 'L2' or `None`. 
	* `solver`: choose loss function (algorithm). different algorithms support different penalties, have different speeds, allowance of parallelization.
		
		***note***: despite the name it's used for classification
4. #### LinearSVC
	* `C`: regularization value.
	* `penalty`: `'l1'` or `'l2'`
	* `loss`: The loss function. The combination of `penalty='l1'` and `loss='hinge'` is not supported. `'squared_hinge'` for `'l2'`
		
		***note***: used for classification
	
**IMPORTANT NOTES**
* for penalties, `l1` reduces some coefficient to 0 while `l2` doesn't. use `'l1'` when you want to select features assuming only some are important. '`l1`' is good for interpretability.
* `C` and `alpha` are opposites. lower `alpha` corresponds to a higher `C`. they are the most important hyperparameters.
* other options are **SGDClassifier** and **SGDRegressor** which implement scalable algorithms.

### Trees
1. #### Decision Tree classifier:
	* `max_depth`: maximum depth of the tree. used to prevent overfitting
	* `max_leaf_nodes`: as the name implies. maximum number of leaf nodes. used to prevent overfiting
	* `min_samples_leaf`: minimum number of samples to become a leaf. used to prevent overfitting.
	* `min_samples_split`: minimum number of samples to split a node. used to prevent overfitting.
	* more *hyperparameter* to apply conditions that change how the tree is constructed. 
	    
		Example:
	    
		  tree = sklearn.tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes=None, min_samples_leaf=None)

		***note***: conditions to prune trees can be set before training (*pre-pruning parameters*) or after training *(post-pruning)*. in sci-kit learn only pre-pruning is implemented.

2. #### RandomForestClassifier:
	* all decision tree *hyperparameters*
	* `n_estimators`: how many decision tree to train
	* `max_features`: very important parameter. the higher the number the less variation among the trees and therefore overfitting is more likely. default is `auto` which is the square root or the number of features. if `1`, trees have no freedom to test and randomness is maximized.
	* `bootstrap`: if true, *n* samples will be selected randomly with replacement from the dataset which contains *n* number of samples. this means the bootstrap dataset is as big as the original but with repeated samples.
	* `max_samples`: only available with bootstrap=True. max number of samples to be pulled from the dataset. default is None.
	* `oob_score`: only available when bootstrap=True. use the remaining samples to test the model.
		
		Example:
		
		  forest = sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_features='sqrt', bootstrap=True, oob_score=True)

		***note***: `n_jobs` parameter is used to parallelize the training of the trees. pass the number of CPU cores to be used.

3. #### GradientBoosting:
	* all decision tree *hyperparameters*
	* `n_estimators`: number of trees in the ensemble. this hyperparameter has a different effect than random forest. more estimators can lead to overfitting
	* `learning_rate`: the rate of contribution of trees. it interact with `n_estimator`
	* `max_features`: default is `None`. can take `auto`, `sqrt`,  `log2`, a fraction or *int*. choosing `max_features` < `n_features` leads to reduction is variace and an increase in bias

**IMPORTANT NOTES**:
* for each algorithm there are more hyperparameters to control than mentioned.
* in scikit-learn, there are two interesting technics in **sklearn.ensemble** module.
	* **VotingClassifier**(Regressor): ensemble different classification(regression) algorithms to cast *votes*. two methods of voting *hard* and *soft*
	*  **StackingClassifier**(Regressor): ensemble different algorithms for boosting.
* *boosting* uses *weak learners* to create a powerful model unlike algorithms where the tree are independent from each other.
* Tree based algorithms doesn't require scaling of features and most of the time little hyperparameters tuning.
* more algorithms such as **AdaBoostClassifier**, **ExtraTreeClassifier**, **IsolationForest** etc.

### Kernelized Support Vector Machines
1. ####  SVC (SupportVectormachinesClassifier)
	* `C`: regularization parameter. the same as in LinearSVC. higher value reduces complexity and a lower value makes the model more sensitive.
	* `kernel`: kernel in support vector machine is the algorithm of transformation into higher dimensional space. `ploy` is limited by the `degree` parameter. `rbf` (radial basis function) considers all possible polynomials.
	*  `gamma` next to C, gamma is the most important parameter in SVM. a higher gamma makes the model more sensitive to individual data points.
	* more kernel specific hyperparameters

		example:

		  svc = sklearn.svm.SVC(kernel='rbf', C=200, gamma=.2)
**IMPORTANT NOTES**:
* SVM often requires hyperparameters tuning.
* unlike tree based algorithms, they require preprocessing of data. normalizing/standardizing of data is required for good performance.
* they don't scale very well with data. they can be computationally expensive.
* produces complex decision boundaries (which is good)

### Neural networks
1. ####  MLPClassifier (Multi-Layer Perceptron)
	* `hidden_layer_sizes`: determines the size of the network. takes a tuple where each index and value corresponds to the layer and its size respectively.
	* `acitivation`: activation function to mitigate linearity. default is `'relu'`. other options are `'tanh'`, `'logisitic'` and `'identity'`.
	* `solver`: weight optimizer algorithm. three optimizers are supported `'adam'`, `'sgd'` and `'ldfbg'`. `'adam'` is the default `'optimizer'`.
	* `alpha`: *L2* penalty regularization parameter (not the one that reduces some coefficients to zero)  
	* `batch_size`: size of mini-batches for stochastic optimizers (`adam` and `sgd`)
	* `learning_rate`: how to handle learning rate. 
		* `constant` is a constant value determined by the parameter `learning_rate_init`.
		* `invscaling` decreases the learning rate gradually. the magnitude of the decrease in determined by `power_t` parameter.
		* `adaptive` as long as the loss is decreasing the learning rate remains constant. each time two consecutive epochs fail in decreasing the cost or increasing the validation score, the current learning rate is divided by `5`.
	* `learning_rate_init`: initial learning rate (or constant in case constant is passed)
	* **many other parameters**

	Example:

	    mlp = sklearn.neural_network.MLPClassifier(
	                   hidden_layer_sizes = [100, 100],
	                   activiation = 'tanh',
	                   solver = 'adam',
	                   alpha = '0.9',
	                   learning_rate_init = 0.0001
	          )
 
 **IMPORTANT NOTES**:
 * requires preprocessing. input features should be 'homogenous'
 * scikit-learn doesn't support the use of GPU and it's not a deep learning library.
 * requires heavy hyperparameters tuning.
 * the neural network is defined by `hidden_layer_sizes`, `alpha` and `activation`. how does it learn is defined by `solver`

## Uncertainty Estimates from Classifiers
In scikit-learn there are two methods to obtain certainty estimates from classifiers. most (not all) classifiers have at least one of them. 


1. #### Classifier.decision_function()

	the method returns an array of shape `(n_samples, n_classes)` with the exception of binary classification where the shape is `(n_samples,)`. zero means complete uncertainty. in case of binary classification, negative values mean that the negative class was decided and vice versa for the positive class. in binary classification, the negative class is the first class in `classifier.classes_`. the magnitude of the value is the magnitude of certainty. interpreting the values might be difficult as the range is quite arbitrary and depends on the classifier and the dataset.

	example:

		[Input]
		# binary classification
		gb = GradientBoostingClassifier().fit(X_tr, y_tr)
		gb.decision_function(X_te[:6])
		
		[Output]
		[ 4.29870147 -2.9459108  -4.42422003 -3.75275711  4.26842451  3.66779101]
		
		[Input]
		(gb.decision_function(X_te[:6]) > 0).astype(int)
		
		[Output]
		[1 0 0 0 1 1]
	if we looked at the sign only we see the class that was the decided. the magnitude is the certainty.
	
2. ####  Classifier.predict_proba()

	it returns an array of shape `(n_samples, n_classes)` for all cases. 2 columns for the probability of each class. probability estimates are easier to understand the decision function
	
	example:
		
	   [Input]
	   gb = GradientBoostingClassifier().fit(X_tr, y_tr)
	   gb.predict_proba(X_te[:3])

	   [Output]
	   [[0.10217718 0.78840034 0.10942248]
	   [0.78347147 0.10936745 0.10716108]
	   [0.09818072 0.11005864 0.79176065]]

	   [Input]
	   np.argmax(gb.predict_proba(X_te[:10]), axis=1)

	   [Output]
	   [1  0  2]

## Unsupervised learning and Preprocessing
all scaling functions have the same interface. use `fit` on the training set to find the scaling parameters and then `transform` to scale the data
### Scalers
1. #### MinMaxScaler
	* scale the data so that the *Min* == 0 and *Max* == 1

		Example:

		  [Input]
		  X
		  
		  [Output]
		  [[3],
		   [6],
		   [9]]
		   
		  [Input]
		  sklearn.preprocessing.MinMaxScaler().fit_transform(X)

		  [Output]
		  [[0. ],
		   [ .5],
		   [1. ]]

2. #### SandardScaler
	* scales so that the *mean* == 0 and *variance* is 1


	Example:
	
	   [Input]
	   X
		  
	   [Output]
	   [[3],
	    [6],
	    [9]]
		   
	   [Input]
	   sklearn.preprocessing.StandardScaler().fit_transform(X)

	   [Output]
	   array([[-1.22474487],
		      [ 0.        ],
		      [ 1.22474487]])

	***note***: standard scaler is very sensitive to outliers and it doesn't scale the data well when you have extreme values.

3. #### RobustScaler
	* similar to StandardScaler but uses the median and quantiles instead of the mean and the variance. in this way it accounts for extreme values.

		Syntax:

		  sklearn.preprocessing.RobustScaler().fit_transform(X)

4. #### Nomalizer
	* it's a very different kind of rescaling. it scales each data point such that the feature vector has a Euclidean length of 1. in other words, if projects a data point on a circle with a radius of 1.

		Syntax:
		
		  sklearn.preproccessing.Normalizer().fit_transform(X)

**IMPORTANT NOTES**:
* testing data should be rescaled using the same scaler that is fitted on the training data. don't fit again. transform with the fitted model.
* RobustScaler works better than StandardScaler with extreme value. it's not sensitive to outliers. 

### Dimensionality reduction
1. #### PCA (Principle Components Analysis)
	for a dataset with a certain number of numeric features, PCA rotates the dimensional space so that it finds the directions that has the most variation. the direction that has the most variation is always the first component, the direction that has the second most variation is the second component and so on. each component is a mix of original features and consequently the components are difficult to interpret. PCA is typically used to reduce the features so that they can be visualized. it can be used for feature engineering also.
	
	important parameters:
	* `n_components`: number of components to retain.
	* `whiten`: whether to rescale the data after the transformation. this is the same as using `SandardScaler` after the transformation.

	example:
	
	   pca = sklearn.decomposition.PCA(n_components=3)
	   X_pc = pca.fit_transform(X)

	important attributes:
	
	* `.components_`: components as vectors of coefficients of original features
	* `.explained_variance_ratio_`: percentage of explained variance by each component retained.

	***note***: there always will be a loss of information when using PCA to reduce dimensionality. full information will be captured only if the number of components are equal to the number of features which means that dimensions were not reduced only transformed.

2. #### NMF (Non-negative Matrix Factorization)
	NMF differs from PCA in that choosing a different `n_components` doesn't just retain components, but a different number completely yields different transformation of the data. unlike PCA the components in NMF has no meaningful order and the components are easier to interpret. NMF is useful to extract original signals from data that is made of of several independent sources.
	* `n_components`: number of components, unlike PCA different number yields different transformation of data.
	
	Example:
		
	   nmf = sklearn.decomposition.NMF(n_components=4)
	   X_nmf = nmf.fit_transform(X)

3. #### TSNE (T-distributed Stochastic Neighbor Embedding)
	t-SNE is a great algorithm that is mainly used for visualization. beside visualization t-SNE has little use as it can't apply learned transformations on different or new data, only the data it was trained with. it's great for EDA. while PCA often does a good job in reducing dimensionality for visualization, the method has it's downsides. t-SNE projects the points in high dimensional space into 2 or 3 dimensions and tries to preserve the distances between data points as much as possible. the closer the points in the high dimensional space the closer they appear in the new representation and the farther they are the farther they are represented.

	* `n_components`: 2 by default. doesn't make sense to choose more than 3 since the main use is visualization.
	* `perplexity`: perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results.
	* `early_exaggeration`: Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. The choice of this parameter is not very critical.

	example:
		
	   tsne = sklearn.manifold.TSNE(perplexity=50, early_exaggeration=12)
	   X_tnse = tsne.fit_transform(X)
	   _ = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)

	***note***: t-SNE is a stochastic algorithm. the values of the points are randomly initialized and they don't converge into specific locations in the lower dimensional space. this means that it's very sensitive to the random state.

