# Models parameters summary (scikit learn)
1. **KNN classification and regression**:
	* `k`: number of neighbors.
	
	example:
	
	      cls = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)
	      reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=4)       
2. **Ridge regression**:
	* `alpha`: coefficients regularization (L2). default is 1. higher value means coefficients are closer to zero. when value is zero it's the same as normal plain linear regression.
	
	example:
	
          ridge = sklearn.linear_model.Ridge(alpha=0.1)
3. **Lasso**:
	* `alpha`: coefficients regularization (L1). reduces certain coefficients to zero (could be considered as automatic feature selection). higher value means a simpler model. default value is 1
	* `max_iter`: maximum number of iterations to run
	
	example:
	
          lasso = sklearn.linear_model.Lasso(alpha=0.01, max_iter=10_000)

4. **LogisticRegression**:
	* `C`: coefficients regularization value. higher value of C corresponds to lower regularization and higher complexity model. this means the higher the value of C, the higher the coefficients.
	* `penalty`: specify the norm of regularization. `'L1'` selects features, `'L2'` reduces all coefficients, `'elasticnet'` mix of 'L1' and 'L2' or `None`. 
	* `solver`: choose loss function (algorithm). different algorithms support different penalties, have different speeds, allowance of parallelization.
	* `max_iter`: max iteration to run
	
          

