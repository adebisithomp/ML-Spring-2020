Red wine dataset: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Install scikit-learn: https://scikit-learn.org/stable/install.html
- Install using the terminal command: pip install -U scikit-learn

To generate all graphs and data, uncomment out the code in sections and run the code. The graphs called with create_plot_single() will automatically be saved to your directory. The graphs called with create_plot() and create_scatter_plot() will be opened up once you run the code. The sections are outlined below. 

Create Scatter plot of data
-create_scatter_plot()

Random Forest (rf)
-random_forest_error(): returns accuracy of random forest w/out cross validation
-random_forest_gs(): returns average test score, best estimator, best parameters, and confidence intervals of nested cross validation and grid search of random forest
-random_forest_cv_error(): returns optimal accuracy, optimal confidence intervals, and cross validation scores for random forest after tuning
-create_plot_single(): returns a plot optimal confidence intervals and optimal CV score

Support Vector Machine (svm)
-svm_error(): returns accuracy of SVM w/out cross validation
-svm_gs(): returns average test score, best estimator, best parameters, and confidence intervals of nested cross validation and grid search of SVM
-svm_cv_error(): returns optimal accuracy, optimal confidence intervals, and cross validation scores for SVM after tuning
-create_plot_single(): returns a plot optimal confidence intervals and optimal CV score

Neural Net (nn)
-neural_net_error(): returns accuracy of a neural net w/out cross validation
-neural_net_gs(): returns average test score, best estimator, best parameters, and confidence intervals of nested cross validation and grid search of a neural net
-neural_net_cv_error(): returns optimal accuracy, optimal confidence intervals, and cross validation scores for a neural net after tuning
-create_plot_single(): returns a plot optimal confidence intervals and optimal CV score

Create plot of all CV score's of models together
-create_plot()