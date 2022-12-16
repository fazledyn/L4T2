"""
main code that you will run
"""
from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score, accuracy


if __name__ == '__main__':
    # data load
    X, y = load_dataset()
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    params = dict()
    params['n_iters'] = 1000
    params['lr'] = 0.01
    n_estimator = 9

    classifier = BaggingClassifier(params, n_estimator=n_estimator)
    classifier.fit(X_train, y_train)
    
    ## testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Performance of Bagging')
    print('Accuracy \t:',       accuracy(       y_true=y_test, y_pred=y_pred))
    print('Recall score \t:',   recall_score(   y_true=y_test, y_pred=y_pred))
    print('Precision score :',  precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score \t:',       f1_score(       y_true=y_test, y_pred=y_pred))

