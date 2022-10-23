import sys
sys.path.append('.')

from utils import train_dev_test_split
from sklearn import datasets
from joblib import dump, load


def test_get_bias():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    label = digits.target
    train_fracs = 0.7
    dev_fracs = 0.1
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_fracs, dev_fracs
    )
    model = load("svm_gamma=0.0008_C=2.0.joblib")
    predict = model.predict(x_test)
    checker = predict[0]
    flag = True
    for item in predict:
        if checker != item:
            flag = False
            break;
    assert flag != True
def test_predict_all_classes():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    label = digits.target
    model = load("svm_gamma=0.0008_C=2.0.joblib")
    predict = model.predict(data)
    checker = list(set(label))
    predicted = list(set(predict))
    assert len(checker) == len(predicted)
    






















