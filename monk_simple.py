import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import pprint as pp
from sklearn import preprocessing
from valentiMLP import ValentiMLP
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=matplotlib.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot(errors):
    i = 0
    colours = ["r", "g", "b", "y", "m", "c", "k"]
    linestyles = ["-", "--", "-.", ":"]

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Error over epochs")
    plt.grid(True)
    
    for key, val in errors.items():
        xs, ys = np.array(val).T
        plt.plot(xs, ys, label=key, c=colours[i], ls=linestyles[i])     
        i += 1

    plt.legend()
    plt.show()

def one_hot_encoding(value, n_classes):
    output = [-.9] * n_classes
    output[value-1] = .9
    return output

# load dataset
df_data_tr = pd.read_csv("./dataset/monk/monks-3.train", sep=" ", index_col=8, header=None)
df_data_ts = pd.read_csv("./dataset/monk/monks-3.test", sep=" ", index_col=8, header=None)

df_target_tr = df_data_tr[1]
df_target_ts = df_data_ts[1]

# remove index and target columns from dataset
df_data_tr.drop([0,1], axis="columns", inplace=True)
df_data_ts.drop([0,1], axis="columns", inplace=True)

# preprocessing: convert dataset features into one-hot encoding.
data_tr = []
for row in df_data_tr.iterrows():
    row = row[1]
    tr_sample = []
    tr_sample += one_hot_encoding(row[2], 3)
    tr_sample += one_hot_encoding(row[3], 3)
    tr_sample += one_hot_encoding(row[4], 2)
    tr_sample += one_hot_encoding(row[5], 3)
    tr_sample += one_hot_encoding(row[6], 4)
    tr_sample += one_hot_encoding(row[7], 2)
    data_tr.append(tr_sample)

data_tr = np.array(data_tr)

data_ts = []
for row in df_data_ts.iterrows():
    row = row[1]
    ts_sample = []
    ts_sample += one_hot_encoding(row[2], 3)
    ts_sample += one_hot_encoding(row[3], 3)
    ts_sample += one_hot_encoding(row[4], 2)
    ts_sample += one_hot_encoding(row[5], 3)
    ts_sample += one_hot_encoding(row[6], 4)
    ts_sample += one_hot_encoding(row[7], 2)
    data_ts.append(ts_sample)

data_ts = np.array(data_ts)

# preprocessing: convert targets into one-hot encoding.
df_target_tr = pd.DataFrame(df_target_tr)
df_target_tr["one_hot"] = df_target_tr[1].apply(lambda el : np.array([.9, -.9]) if el == 0 else np.array([-.9, .9]) )

df_target_ts = pd.DataFrame(df_target_ts)
df_target_ts["one_hot"] = df_target_ts[1].apply(lambda el : np.array([.9, -.9]) if el == 0 else np.array([-.9, .9]) )

target_tr = np.array(df_target_tr["one_hot"].values.tolist())
target_ts =  np.array(df_target_ts["one_hot"].values.tolist())

# Default initaial parameters (not affected by model selection).
default_params = {
    "task" : "classification",
    "n_input" : 17,
    "n_output": 2,
    "n_epoch" : 2000,
}

# Model selection: selecting best parameters.
param_grid = {
    "n_hidden" : [2, 3, 4, 5],
    "eta"      : [0.0001, 0.001, 0.01],
    "alpha"    : [0.1, 0.3, 0.5, 0.7, 0.9],
    "lambda_"  : [1e-5, 1e-4, 1e-3],
}

X_train = data_tr
y_train = target_tr
X_test  = data_ts
y_test  = target_ts

# initialise model 
n_int_fold = 10 # number of folds
model = ValentiMLP(**default_params, n_batch=int(len(data_tr)/n_int_fold))

# grid search
grid_search = GridSearchCV(model, cv=n_int_fold, n_jobs=-1, param_grid=param_grid, verbose=2, return_train_score=True)
grid_search.fit(X_train, y_train)

# print results
cv_result = pd.DataFrame(grid_search.cv_results_)
pprinter = pp.PrettyPrinter(indent=4)
print(cv_result[["param_eta","param_alpha", "param_lambda_", "param_n_hidden", "mean_train_score","std_train_score","mean_test_score","std_test_score"]])
print("Best parameters:")
pprinter.pprint(grid_search.best_params_)

# refit model over whole training set
print("Refit model with best parameters:\n", grid_search.best_params_)
model = ValentiMLP(**default_params, n_batch=len(X_train))
model.set_params(**grid_search.best_params_)
    
model.fit_with_validation(X_train, y_train, X_test, y_test)   

# Measuring accuracy over test set
y_true = [ np.argmax(x) for x in target_ts ]
y_pred = [ np.argmax(x) for x in model.predict(data_ts) ]

print("Best parameters:")
pprinter.pprint(grid_search.best_params_)
print("- Classification report: \n",metrics.classification_report(y_true, y_pred))

print("- Error (TR): ", model.score(X_train, y_train))
print("- Error (TS): ", model.score(X_test, y_test))
print("- Accuracy (TR): ", model.accuracy(X_train, y_train))
print("- Accuracy (TS): ", model.accuracy(X_test, y_test))

print("- Accuracy 1 (TS): ", metrics.accuracy_score(y_true, y_pred))

# Compute confusion matrix
class_names = ["True", "False"]
cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

plot({
    "MSE TR"      : model.scores_tr_,
    "MSE TS"     : model.scores_val_,
})

plot({
    "ACCURACY TR" : model.accs_tr_,
    "ACCURACY TS": model.accs_val_
})