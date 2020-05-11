import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from scipy import stats
import time



def get_test_train(fname,seed,datatype):
	data = np.genfromtxt(fname,delimiter=';',dtype=datatype)
	data = np.delete(data, (0), axis=0)
	X = data[:,:-1]
	y = data[:,-1].reshape(-1,1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
	return X_train, X_test, y_train, y_test, X, y

def load_wine(path=''):
	return get_test_train(os.path.join(path,'winequality-red.csv'),seed=1567708903,datatype=float)

def create_scatter_plot(x_data, x_label, y_data, y_label, title):
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.scatter(x_data[:,0], x_data[:,1], c = y_data.ravel())
	plt.title(title)
	plt.show()

def create_plot(score_1, score_2, score_3, title):
	plt.plot(score_1, label = "Random Forest")
	plt.plot(score_2, label = "SVM")
	plt.plot(score_3, label = "Neural Net")
	plt.xlabel("Folds")
	plt.ylabel("CV Score")
	plt.legend(loc = 'lower right')
	plt.ylim([.4,.8])
	plt.grid()
	plt.title(title)
	plt.show()

def create_plot_single(scores, title, save):
	plt.plot(scores)
	plt.plot(scores + (scores.std()*2))
	plt.plot(scores - (scores.std()*2))
	plt.xlabel("Folds")
	plt.ylabel("CV Score")
	plt.ylim([.4,.8])
	plt.grid()
	plt.title(title)
	plt.savefig(save)
	plt.clf()

def cofidence_interval(a, confidence):
	mean, sigma = np.mean(a), np.std(a)
	conf_int = stats.norm.interval(confidence, loc=mean, scale=sigma / np.sqrt(len(a)))
	return conf_int

class Random_Forest():

	def random_forest_error(self, train_X, train_Y, test_X, test_Y, n_estimators, max_depth):
		clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, random_state = 0)
		clf.fit(train_X, train_Y.ravel())
		y_pred = clf.predict(test_X)
		accuracy = accuracy_score(test_Y, y_pred)
	
		return accuracy

	def random_forest_gs(self, X, Y, scoring, cv):
		rf = RandomForestClassifier(random_state = 0)
		grid_params = {
		"n_estimators": np.array(range(94,109)), 
		"max_depth": np.array(range(11,17))
		}

		inner_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		outer_cv = KFold(n_splits= cv, shuffle=True, random_state=0)

		gs = GridSearchCV(
		estimator = rf,
		param_grid= grid_params,
		scoring = scoring,
		cv = inner_cv,
		n_jobs = -1)
		gs.fit(X, Y.ravel())
		best_params = gs.best_params_
		best_estimator = gs.best_estimator_

		nested_cv = cross_validate(gs, X=X, y=Y.ravel(), cv=outer_cv, return_train_score=True)
		mean_val_score = nested_cv['test_score'].mean()
		conf_int = cofidence_interval(nested_cv['test_score'], 0.95)
		return mean_val_score, best_estimator, best_params, conf_int


	def random_forest_cv_error(self, test_X, test_Y, n_estimators, max_depth, cv, scoring):
		clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, random_state = 0)
		calc_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		scores = cross_val_score(clf, test_X, test_Y.ravel(), scoring = scoring, cv = calc_cv)
		conf_int = cofidence_interval(scores, 0.95)
		return scores.mean(), conf_int, scores 


class SVM():

	def svm_error(self, train_X, train_Y, test_X, test_Y, C, gamma):
		clf = SVC(C = C, kernel = 'rbf', gamma = gamma)
		clf.fit(train_X, train_Y.ravel())
		y_pred = clf.predict(test_X)
		accuracy = accuracy_score(test_Y, y_pred)
		return accuracy

	def svm_gs(self, X, Y, scoring, cv):
		grid_params =  {'C': np.array([200,250,300,320,350]), 
		'gamma': np.array([0.001, 0.01])}

		svm = SVC(kernel="rbf")

		inner_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		outer_cv = KFold(n_splits= cv, shuffle=True, random_state=0)

		gs = GridSearchCV(
		estimator = svm,
		param_grid= grid_params,
		scoring = scoring,
		cv = inner_cv,
		n_jobs = -1)
		gs.fit(X, Y.ravel())
		best_params = gs.best_params_
		best_estimator = gs.best_estimator_

		nested_cv = cross_validate(gs, X=X, y=Y.ravel(), cv=outer_cv, return_train_score=True)
		mean_val_score = nested_cv['test_score'].mean()
		conf_int = cofidence_interval(nested_cv['test_score'], 0.95)
		return mean_val_score, best_estimator, best_params, conf_int

	def svm_cv_error(self, test_X, test_Y, C, gamma, cv, scoring):
		clf = SVC(C = C, kernel = 'rbf', gamma = gamma)
		calc_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		clf.fit(test_X, test_Y.ravel())
		scores = cross_val_score(clf, test_X, test_Y.ravel(), scoring = scoring, cv = calc_cv)
		conf_int = cofidence_interval(scores, 0.95)
		return scores.mean(), conf_int, scores


class Neural_Net():

	def neural_net_error(self, train_X, train_Y, test_X, test_Y, hidden_layer_sizes, max_iter, alpha):
		clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs', alpha=alpha, 
			random_state= 0,
			max_iter= max_iter)
		clf.fit(train_X, train_Y.ravel())
		y_pred = clf.predict(test_X)
		accuracy = accuracy_score(test_Y, y_pred)
		return accuracy

	def neural_net_gs(self, X, Y, scoring, cv):
		mlp = MLPClassifier(solver = 'lbfgs', random_state = 0 )
		grid_params = {  
		'max_iter': [800,900,1000,1110],
		'alpha': [0.001, 0.01, 0.1], 
		'hidden_layer_sizes': [(15,15,15,15), (15,15,15,15,15)]
		}

		inner_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		outer_cv = KFold(n_splits= cv, shuffle=True, random_state=0)


		gs = GridSearchCV(
		estimator = mlp,
		param_grid= grid_params,
		scoring = scoring,
		cv = inner_cv,
		n_jobs = -1)
		gs.fit(X, Y.ravel())
		best_params = gs.best_params_
		best_estimator = gs.best_estimator_

		nested_cv = cross_validate(gs, X=X, y=Y.ravel(), cv=outer_cv, return_train_score=True)
		mean_val_score = nested_cv['test_score'].mean()
		conf_int = cofidence_interval(nested_cv['test_score'], 0.95)
		return mean_val_score, best_estimator, best_params, conf_int 

	def neural_net_cv_error(self, test_X, test_Y, hidden_layer_sizes, alpha, max_iter, cv, scoring):
		clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs', alpha=alpha, 
			random_state= 0,
			max_iter= max_iter)
		calc_cv = KFold(n_splits= cv, shuffle=True, random_state=0)
		scores = cross_val_score(clf, test_X, test_Y.ravel(), scoring = scoring, cv = calc_cv)
		conf_int = cofidence_interval(scores, 0.95)
		return scores.mean(), conf_int, scores


def main():

	wine_data = load_wine()
	train_X, train_Y = wine_data[0], wine_data[2]
	test_X, test_Y = wine_data[1], wine_data[3]

	start_time = time.time()

	scoring = 'accuracy'

	# create_scatter_plot(x_data = X_wine, x_label = "Fixed Acidity", y_data = y_wine,
	#  y_label = "Volatile Acidity" , title = "Fixed vs Volatile Acidity")


	rf = Random_Forest()
	# rf_accuracy = rf.random_forest_error(train_X, train_Y, test_X, test_Y, n_estimators = 82, max_depth = 14)
	# print("Random Forest Accuracy: ", rf_accuracy)

	# rf_best_score, rf_best_estimator, rf_best_params, rf_conf_int = rf.random_forest_gs(train_X, 
	# train_Y, scoring = scoring, cv = 5)
	# print(rf_best_score)
	# print(rf_best_estimator)
	# print(rf_best_params)
	# print(rf_conf_int)

	rf_optimal_accuracy, rf_optimal_conf_inf, rf_scores = rf.random_forest_cv_error(test_X, test_Y,
	 n_estimators = 82, max_depth = 14, cv = 5, scoring = scoring)
	print(rf_optimal_accuracy)
	print(rf_optimal_conf_inf)

	create_plot_single(rf_scores, "Random Forest w/Confidence Intervals", "rf.png")

	svm = SVM()
	# svm_accuracy = svm.svm_error(train_X, train_Y, test_X, test_Y, C = 1.0)
	# print("SVM Accuracy: ", svm_accuracy)

	# svm_best_score, svm_best_estimator, svm_best_params, svm_conf_int = svm.svm_gs(train_X, train_Y, 
	# 	scoring = scoring, cv = 5)
	# print(svm_best_score)
	# print(svm_best_estimator)
	# print(svm_best_params)
	# print(svm_conf_int)

	svm_optimal_accuracy, svm_optimal_conf_inf, svm_scores = svm.svm_cv_error(test_X, test_Y,
	 C = 300, gamma = 0.001, cv = 5, scoring = scoring)
	print(svm_optimal_accuracy)
	print(svm_optimal_conf_inf)

	create_plot_single(svm_scores, "SVM w/Confidence Intervals", "svm.png")


	nn = Neural_Net()
	# nn_accuracy = nn.neural_net_error(train_X, train_Y, test_X, test_Y,
	#  hidden_layer_sizes = (20,20,20), max_iter = 5000)
	# print("Neural Net Accuracy: ", nn_accuracy)

	# nn_best_score, nn_best_estimator, nn_best_params, nn_conf_int = nn.neural_net_gs(train_X, 
	# train_Y, scoring = scoring, cv = 5)
	# print(nn_best_score)
	# print(nn_best_estimator)
	# print(nn_best_params)
	# print(nn_conf_int)

	nn_optimal_accuracy, nn_optimal_conf_inf, nn_scores = nn.neural_net_cv_error(test_X, test_Y, 
		hidden_layer_sizes = (15,15,15,15), alpha = 0.001, max_iter = 1000, cv = 5, scoring = scoring)
	print(nn_optimal_accuracy)
	print(nn_optimal_conf_inf)

	create_plot_single(nn_scores, "Neural Net w/Confidence Intervals", "nn.png")

	create_plot(rf_scores, svm_scores, nn_scores, "All Classifications CV Scores")

	print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
	main()