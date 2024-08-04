#Import svm model
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


def train_svm(X_train, y_train, kernel='poly'):
    #Create a svm Classifier
    clf = svm.SVC(kernel=kernel)

    #Train the model using the training sets
    clf.fit(X_train, y_train)
    return clf

def evaluate_svm(clf, X_test, y_test):
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
