import pickle

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

with open('X_train_lbp', 'rb') as f:
    X_train_lbp_2 = pickle.load(f)
    
with open('X_test_lbp', 'rb') as f:
    X_test_lbp_2 = pickle.load(f)
    
with open('Dataset', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f) 

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

grid = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
grid.fit(X_train_lbp_2, y_train)

scores = cross_val_score(grid.best_estimator_, X_train_lbp_2, y_train, cv=5, scoring='accuracy')
print(scores)

grid.best_params_

predictions = grid.best_estimator_.predict(X_test_lbp_2)
print(accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))
