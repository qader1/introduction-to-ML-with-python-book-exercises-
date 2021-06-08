# sklearn summary
## Supervised learning
### 1. **KNN classification and regression**:
* `k`: number of neighbors.
	
example:
	
	cls = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)
	reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=4)
### Linear Models  
* `alpha`: coefficients regularization (L2). default is 1. higher value means coefficients are closer to zero. when value is zero it's the same as normal plain linear regression.
	
	example:
	
      ridge = sklearn.linear_model.Ridge(alpha=0.1)

#### 2. **Lasso**:
* `alpha`: coefficients regularization (L1). reduces certain coefficients to zero (could be considered as automatic feature selection). higher value means a simpler model. default value is 1

	example:
	
	  lasso = sklearn.linear_model.Lasso(alpha=0.01, max_iter=10_000)

#### 3. **LogisticRegression**:
* `C`: coefficients regularization value. higher value of C corresponds to lower regularization and higher complexity model. this means the higher the value of C, the higher the coefficients.
* `penalty`: specify the norm of regularization. `'l1'` selects features, `'l2'` reduces all coefficients, `'elasticnet'` mix of 'L1' and 'L2' or `None`. 
* `solver`: choose loss function (algorithm). different algorithms support different penalties, have different speeds, allowance of parallelization.
	
	***note***: despite the name it's used for classification
#### 4. **LinearSVC**
* `C`: regularization value.
* `penalty`: `'l1'` or `'l2'`
* `loss`: The loss function. The combination of `penalty='l1'` and `loss='hinge'` is not supported. `'squared_hinge'` for `'l2'`
	
	***note***: used for classification
	
**IMPORTANT NOTES**
* for penalties, `l1` is used reduces some coefficient to 0 while `l2` doesn't. use `'l1'` when you want to select features assuming only some are important. '`l1`' is good for interpretability.
* `C` and `alpha` are opposites. lower `alpha` corresponds to a higher `C`. they are the most important hyperparameters.
* other options are **SGDClassifier** and **SGDRegressor** which implement scalable algorithms.

### Trees
#### 1. **Decision Tree classifier**:
* `max_depth`: maximum depth of the tree. used to prevent overfitting
* `max_leaf_nodes`: as the name implies. maximum number of leaf nodes. used to prevent overfiting
* `min_samples_leaf`: minimum number of samples to become a leaf. used to prevent overfitting.
* `min_samples_split`: minimum number of samples to split a node. used to prevent overfitting.
* more *hyperparameter* to apply conditions that change how the tree is constructed. 
    
	Example:
    
	  tree = sklearn.tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes=None, min_samples_leaf=None)

	***note***: conditions to prune trees can be set before training (*pre-pruning parameters*) or after training *(post-pruning)*. in sci-kit learn only pre-pruning is implemented.

#### 2. **RandomForestClassifier**:
* all decision tree *hyperparameters*
* `n_estimators`: how many decision tree to train
* `max_features`: very important parameter. the higher the number the less variation among the trees and therefore overfitting is more likely. default is `auto` which is the square root or the number of features. if `1`, trees have no freedom to test and randomness is maximized.
* `bootstrap`: if true, *n* samples will be selected randomly with replacement from the dataset which contains *n* number of samples. this means the bootstrap dataset is as big as the original but with repeated samples.
* `max_samples`: only available with bootstrap=True. max number of samples to be pulled from the dataset. default is None.
* `oob_score`: only available when bootstrap=True. use the remaining samples to test the model.
	
	Example:
	
	  forest = sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_features='sqrt', bootstrap=True, oob_score=True)

	***note***: `n_jobs` parameter is used to parallelize the training of the trees. pass the number of CPU cores to be used.

#### 3. **GradientBoosting**:
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
#### 1. SVC (SupportVectormachinesClassifier)
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
#### 1. **MLPClassifier (Multi-Layer Perceptron)**
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


          

