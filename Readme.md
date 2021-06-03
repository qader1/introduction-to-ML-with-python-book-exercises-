# Summary of Algorithms
1. **KNN classification and regression**:
	* `k`: number of neighbors.
	
	example:
	
	      cls = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)
	      reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=4)     
## Linear Models  
1. **Ridge regression**:
	* `alpha`: coefficients regularization (L2). default is 1. higher value means coefficients are closer to zero. when value is zero it's the same as normal plain linear regression.
	
	example:
	
          ridge = sklearn.linear_model.Ridge(alpha=0.1)
2. **Lasso**:
	* `alpha`: coefficients regularization (L1). reduces certain coefficients to zero (could be considered as automatic feature selection). higher value means a simpler model. default value is 1
	
	example:
	
          lasso = sklearn.linear_model.Lasso(alpha=0.01, max_iter=10_000)

3. **LogisticRegression**:
	* `C`: coefficients regularization value. higher value of C corresponds to lower regularization and higher complexity model. this means the higher the value of C, the higher the coefficients.
	* `penalty`: specify the norm of regularization. `'l1'` selects features, `'l2'` reduces all coefficients, `'elasticnet'` mix of 'L1' and 'L2' or `None`. 
	* `solver`: choose loss function (algorithm). different algorithms support different penalties, have different speeds, allowance of parallelization.
	
	***note***: despite the name it's used for classification
4. **LinearSVC**
	* `C`: regularization value.
	* `penalty`: `'l1'` or `'l2'`
	* `loss`: The loss function. The combination of `penalty='l1'` and `loss='hinge'` is not supported. `'squared_hinge'` for `'l2'`
	
	***note***: used for classification
	
-- other options are **SGDClassifier** and **SGDRegressor** which implement scalable algorithms.
	
**IMPORTANT NOTES**
* for penalties, `l1` is used reduces some coefficient to 0 while `l2` doesn't. use `'l1'` when you want to select features assuming only some are important. '`l1`' is good for interpretability.
* `C` and `alpha` are opposites. lower `alpha` corresponds to a higher `C`. they are the most important hyperparameters.

## Trees
1. **Decision Tree classifier**:
    * `max_depth`: maximum depth of the tree. used to prevent overfitting
    * `max_leaf_nodes`: as the name implies. maximum number of leaf nodes. used to prevent overfiting
    * `min_samples_leaf`: minimum number of samples to become a leaf. used to prevent overfitting.
    * `min_samples_split`: minimum number of samples to split a node. used to prevent overfitting.
    * more *hyperparameter* to apply conditions that change how the tree is constructed. 
    
    Example:
    
		tree = sklearn.tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes=None, min_samples_leaf=None)

	***note***: conditions to prune trees can be set before training (*pre-pruning parameters*) or after training *(post-pruning)*. in sci-kit learn only pre-pruning is implemented.

2. **RandomForestClassifier**:
	* all decision tree *hyperparameters*
	* `n_estimators`: how many decision tree to train
	* `max_features`: very important parameter. the higher the number the less variation among the trees and therefore overfitting is more likely. default is `auto` which is the square root or the number of features. if `1`, trees have no freedom to test and randomness is maximized.
	* `bootstrap`: if true, *n* samples will be selected randomly with replacement from the dataset which contains *n* number of samples. this means the bootstrap dataset is as big as the original but with repeated samples.
	* `max_samples`: only available with bootstrap=True. max number of samples to be pulled from the dataset. default is None.
	* `oob_score`: only available when bootstrap=True. use the remaining samples to test the model.
	
	Example:
	
		forest = sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_features='sqrt', bootstrap=True, oob_score=True)

	***note***: `n_jobs` parameter is used to parallelize the training of the trees. pass the number of CPU cores to be used.

3. **GradientBoosting**
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
* more algorithms such as **AdaBoostClassifier**, **ExtraTreeClassifier**, **IsolationForest** etc.
	

	
          

