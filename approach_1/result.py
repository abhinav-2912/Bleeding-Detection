import pickle

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

with open('X_train_sparse_code', 'rb') as f:
    X_train_sparse_code_5 = pickle.load(f)
    
with open('X_test_sparse_code', 'rb') as f:
    X_test_sparse_code_5 = pickle.load(f)
    
with open('Dataset', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f) 

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10]}]

grid = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='f1_macro')
grid.fit(X_train_sparse_code_5, y_train)


scores = cross_val_score(grid.best_estimator_, X_train_sparse_code_5, y_train, cv=5, scoring='f1_macro')
scores

predictions = grid.best_estimator_.predict(X_test_sparse_code_5)
accuracy_score(y_test, predictions)

print(classification_report(y_test, predictions))
