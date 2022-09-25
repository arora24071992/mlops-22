import matplotlib .pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from beautifultable import BeautifulTable

digits = datasets.load_digits()

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

param_list=[]

n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

x_train, x_dev_test, y_train, y_dev_test = train_test_split(data, digits.target, test_size=1-train_frac, shuffle=True)


x_test, x_dev, y_test, y_dev = train_test_split(data, digits.target, test_size=(dev_frac/(test_frac + dev_frac)), shuffle=True)

GAMMA = [0.01, 0.005]
C = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
table_1 = BeautifulTable()
table_2 = BeautifulTable()
table_1.column_headers = ["", "Train","Test","Dev"]
table_2.column_headers = ["", "Min","Max","Mean", "Median"]

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

  param_list.append(acc)
  param_list.append(acc_test)
  param_list.append(acc_train)

  table_1.append_row([curr,  acc_train,  acc_test,  best_acc])

  #Calculating mean, median, min, max
  maxnum = max(param_list)
  minnum = min(param_list)

  #MEAN
  meancol = sumcol/int(len(param_list))

  #Median
  sorted_second_column = sorted(param_list)

  col_length = len(param_list)
  index = (col_length - 1) // 2
  if col_length % 2 == 0:
      mediancol = ((param_list[index] + param_list[index + 1])/2.0)
  else:
      mediancol = param_list[index]

  table_2.append_row([curr,  maxnum,  minnum,  meancol, mediancol])


print(table_1)
print(table_2)

#Get test set prediction
predicted_test = best_model.predict(x_test)
acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true = y_test)
# print(
#     f"Classification metric: {best_model}:\n"
#     f"{metrics.classification_report(y_test,predicted_test)}\n"
# )

predicted_train = best_model.predict(x_train)
acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true = y_train)
# print(
#     f"Classification metric: {best_model}:\n"
#     f"{metrics.classification_report(y_train,predicted_train)}\n"
# )

#Best Hyperparamater and accuracy

table = BeautifulTable()
table.column_headers = ["", "Train","Test","Dev"]
table.append_row([best_param,  acc_train,  acc_test,  best_acc])
print(table)
