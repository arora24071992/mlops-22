import matplotlib .pyplot as plt
from sklearn import datasets, svm, metrics
from beautifultable import BeautifulTable
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()

#Size of each image
print(str(len(digits.images[0]))+" X "+ str(len(digits.images[0][0])))


def svm_train(dataset):
  train_frac = 0.8
  test_frac = 0.1
  dev_frac = 0.1

  n_samples = len(dataset)
  data = dataset.reshape(n_samples, -1)

  x_train, x_dev_test, y_train, y_dev_test = train_test_split(data, digits.target, test_size=1-train_frac, shuffle=True)

  x_test, x_dev, y_test, y_dev = train_test_split(data, digits.target, test_size=(dev_frac/(test_frac + dev_frac)), shuffle=True)

  GAMMA = [0.01, 0.005, 0.001, 0.0005, 0.0001]
  C = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
  table_1 = BeautifulTable()
  table_1.column_headers = ["", "Train","Test","Dev"]

  h_param_comb = [{'gamma':g,'C':c} for g in GAMMA for c in C]
  assert len(h_param_comb) == len(GAMMA)*len(C)

  best_acc = 0.0
  best_param = None
  best_model = None

  for curr in h_param_comb:
    hyper_params = curr
    clf = svm.SVC()
    clf.set_params(**hyper_params)
    
    clf.fit(x_train,y_train)

    #Get dev set prediction
    predicted_dev = clf.predict(x_dev)
    #Dev set accuracy
    acc = metrics.accuracy_score(y_pred=predicted_dev, y_true = y_dev)

    #Finding the best accuracy
    if(acc > best_acc):
      best_acc = acc
      best_model = clf
      best_param = curr

    #Test set prediction and accuracy
    predicted_test = clf.predict(x_test)
    acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true = y_test)

    #Train set prediction and accuracy
    predicted_train = clf.predict(x_train)
    acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true = y_train)
      
    table_1.append_row([curr,  acc_train,  acc_test,  best_acc])


  print(table_1)

  table = BeautifulTable()
  table.column_headers = ["", "Train","Test","Dev"]
  table.append_row([best_param,  acc_train,  acc_test,  best_acc])
  print(table)

arr_1 =  np.ndarray(shape = (1797,10,10))
for i in range(len(digits.images)):
  new_image = digits.images[i].copy()
  new_image.resize((10,10))
  np.append(arr_1,new_image)
# print(digits.images.shape)
# print(len(digits.images))
# print(len(digits.images[0]))
arr_2 = np.ndarray(shape = (1797,20,20))
for i in range(len(digits.images)):
  new_image = digits.images[i].copy()
  new_image.resize((20,20))
  np.append(arr_2,new_image)

# print(arr_2.shape)
# print(arr_2[0])

arr_3 =  np.ndarray(shape = (1797,30,30))
for i in range(len(digits.images)):
  new_image = digits.images[i].copy()
  new_image.resize((30,30))
  np.append(arr_3,new_image)

print(digits.target)
#svm_train(arr_1)
#svm_train(arr_2)
svm_train(arr_3)

from sklearn.model_selection import train_test_split
from beautifultable import BeautifulTable

def svm_train(dataset):
  train_frac = 0.8
  test_frac = 0.1
  dev_frac = 0.1

  n_samples = len(dataset)
  data = dataset.reshape(n_samples, -1)

  x_train, x_dev_test, y_train, y_dev_test = train_test_split(data, digits.target, test_size=1-train_frac, shuffle=True)

  x_test, x_dev, y_test, y_dev = train_test_split(data, digits.target, test_size=(dev_frac/(test_frac + dev_frac)), shuffle=True)

  GAMMA = [0.01, 0.005, 0.001, 0.0005, 0.0001]
  C = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
  table_1 = BeautifulTable()
  table_1.column_headers = ["", "Train","Test","Dev"]

  h_param_comb = [{'gamma':g,'C':c} for g in GAMMA for c in C]
  assert len(h_param_comb) == len(GAMMA)*len(C)

  best_acc = 0.0
  best_param = None
  best_model = None

  for curr in h_param_comb:
    hyper_params = curr
    clf = svm.SVC()
    clf.set_params(**hyper_params)
    
    clf.fit(x_train,y_train)

    #Get dev set prediction
    predicted_dev = clf.predict(x_dev)
    #Dev set accuracy
    acc = metrics.accuracy_score(y_pred=predicted_dev, y_true = y_dev)

    #Finding the best accuracy
    if(acc > best_acc):
      best_acc = acc
      best_model = clf
      best_param = curr

    #Test set prediction and accuracy
    predicted_test = clf.predict(x_test)
    acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true = y_test)

    #Train set prediction and accuracy
    predicted_train = clf.predict(x_train)
    acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true = y_train)
      
    table_1.append_row([curr,  acc_train,  acc_test,  best_acc])


  print(table_1)

  table = BeautifulTable()
  table.column_headers = ["", "Train","Test","Dev"]
  table.append_row([best_param,  acc_train,  acc_test,  best_acc])
  print(table)
