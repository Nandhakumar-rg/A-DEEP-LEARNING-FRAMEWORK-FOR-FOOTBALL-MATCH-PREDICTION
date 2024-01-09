from uuid import RFC_4122
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score ,auc
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score
from os import path, makedirs, walk
from joblib import dump, load
import json

# Utility Functions


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Model trained in {:2f} seconds".format(end-start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made Predictions in {:2f} seconds".format(end-start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return f1_score(target, y_pred, average='micro'), acc


def model(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))

    # f1, acc = predict_labels(clf, X_test, y_test)
    # print("Test Metrics:")
    # print("-" * 20)
    # print("F1 Score:{}".format(f1))
    # print("Accuracy:{}".format(acc))


def derive_clean_sheet(src):
    arr = []
    n_rows = src.shape[0]

    for data in range(n_rows):

        #[HTHG, HTAG]
        values = src.iloc[data].values
        cs = [0, 0]

        if values[0] == 0:
            cs[1] = 1

        if values[1] == 0:
            cs[0] = 1

        arr.append(cs)

    return arr


# Data gathering

en_data_folder = 'english-premier-league_zip'


# data_folders = [es_data_folder]
data_folders = [en_data_folder]

season_range = (9, 18)

data_files = []
for data_folder in data_folders:
    for season in range(season_range[0], season_range[1] + 1):
        data_files.append(
            'data/{}/data/season-{:02d}{:02d}_csv.csv'.format(data_folder, season, season + 1))

data_frames = []

for data_file in data_files:
    if path.exists(data_file):
        data_frames.append(pd.read_csv(data_file))

data = pd.concat(data_frames).reset_index()
print(data)

# Pre processing

input_filter = ['home_encoded', 'away_encoded', 'HTHG', 'HTAG', 'HS',
                'AS', 'HST', 'AST', 'HR', 'AR']
output_filter = ['FTR']

cols_to_consider = input_filter + output_filter

encoder = LabelEncoder()
home_encoded = encoder.fit_transform(data['HomeTeam'])
home_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['home_encoded'] = home_encoded

encoder = LabelEncoder()
away_encoded = encoder.fit_transform(data['AwayTeam'])
away_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['away_encoded'] = away_encoded

# Deriving Clean Sheet
# htg_df = data[['HTHG', 'HTAG']]
# cs_data = derive_clean_sheet(htg_df)
# cs_df = pd.DataFrame(cs_data, columns=['HTCS', 'ATCS'])

# data = pd.concat([data, cs_df], axis=1)

data = data[cols_to_consider]

print(data[data.isna().any(axis=1)])
data = data.dropna(axis=0)

# Training & Testing

X = data[input_filter]
Y = data['FTR']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

svc_classifier = SVC(random_state=100, kernel='rbf')
lr_classifier = LogisticRegression(multi_class='ovr', max_iter=500)
nbClassifier = GaussianNB()
rfClassifier = RandomForestClassifier()

# print("Support Vector Machine")
# print("-" * 20)
# model(svc_classifier, X_train, Y_train, X_test, Y_test)

print()
print("Logistic Regression one vs All Classifier")
print("-" * 20)
model(lr_classifier, X_train, Y_train, X_test, Y_test)

print()
print("Gaussain Naive Bayes Classifier")
print("-" * 20)
model(nbClassifier, X_train, Y_train, X_test, Y_test)

# print()
# print("Decision Tree Classifier")
# print("-" * 20)
# model(dtClassifier, X_train, Y_train, X_test, Y_test)

print()
print("Random Forest Classifier")
print("-" * 20)
model(rfClassifier, X_train, Y_train, X_test, Y_test)
#roc curve------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
label_encoder.fit(Y)
y=label_encoder.transform(Y)
classes=label_encoder.classes_

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler()
X_train_norm=min_max_scaler.fit_transform(X_train)
X_test_norm=min_max_scaler.fit_transform(X_test)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc

RF=OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
RF.fit(X_train_norm,y_train)
y_pred =RF.predict(X_test_norm)
pred_prob = RF.predict_proba(X_test_norm)

from sklearn.preprocessing import label_binarize
#binarize the y_values

y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plotting    
    plt.plot(fpr[i], tpr[i], linestyle='--', 
             label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))

plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='lower right')
plt.show()
#---------------------------------------------------------------------------
# Exporting the Model
print()
print()
shouldExport = input('Do you want to export the model(s) (y / n) ? ')
if shouldExport.strip().lower() == 'y':
    exportedModelsPath = 'exportedModels'

    makedirs(exportedModelsPath, exist_ok=True)

    dump(lr_classifier, f'{exportedModelsPath}/lr_classifier.model')
    dump(nbClassifier, f'{exportedModelsPath}/nb_classifier.model')
    dump(rfClassifier, f'{exportedModelsPath}/rf_classifier.model')

    exportMetaData = dict()
    exportMetaData['home_teams'] = home_encoded_mapping
    exportMetaData['away_teams'] = away_encoded_mapping

    exportMetaDataFile = open(f'{exportedModelsPath}/metaData.json', 'w')
    json.dump(exportMetaData, exportMetaDataFile)

    print(f'Model(s) exported successfully to {exportedModelsPath}/')
