#==============
#inport modules, components, etc ...
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from collections import Counter
from imblearn.over_sampling import SMOTE

#=============================================
#load the CSV file and print and plot
#some data to make sure the load went ok...
fpath='C:/Users/Eric Sakk/Documents/PythonProjVScode/'
ftype='.csv'
fspec='creditcard'
fname=fpath+fspec+ftype

with open(fname, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = np.array(list(reader)).astype(float)
    
print(headers)
print(data.shape)   
print(data[0:5,0:7])
#Note: once loaded
#      Column 0 = time between transactions
#      Column 30 = Classifcation
nnf=sum(data[:,30]==0)
nf=sum(data[:,30]==1)
print('Number of non-fraudulent transacrions =',nnf)
print('Number of fraudulent transacrions =',nf)
if (nnf>10*nf):
    print('Data set is extremely unbalanced')

#======================================================
# Plot some data ...
plt.plot(data[:, 0],data[:, 1] )
plt.xlabel(headers[0])
plt.ylabel(headers[1])
plt.show()

plt.figure(2)  
plt.title("Outlier detection")
plt.scatter(data[:, 1], data[:, 2], color='black')
plt.show()

#============================================================
#Supervised Learning: 
# Create input data and output classification matrices
input_data=data[:,1:29]    #leave out transaction time differences for now
class_target=data[:,30]

print('Input data matrix size=',input_data.shape) 
print('Output classification matrix size=',class_target.shape) 

#======================================================
#outlier detection ...

#============================================================
#Data preprocessing approaches: scaling, normalization ...
#Compare different to techniques to study effects on the classifier algorithm
#1. Scale the input features to be scaled within 0 and 1
scaler = preprocessing.MinMaxScaler()
scaler.fit(input_data)
scaled_input_data=scaler.transform(input_data)
print('Min vals of normalized data', scaled_input_data.min(axis=0))
print('Max vals of normalized data', scaled_input_data.max(axis=0))


#=========================================================
#Deal with unbalanced set in class 1 ...
#Use Counter method to reiterate the imbalance
class_counter = Counter(class_target)
print('Class count information', class_counter)
smote_fix = SMOTE()
sm_data, sm_target = smote_fix.fit_resample(scaled_input_data, class_target)
sm_class_counter = Counter(sm_target)
print('Class count information after SMOTE', sm_class_counter)

np.savez_compressed(fpath+'smote_data_targets.npz', sm_data,sm_target)


#=======================================================
"""
#create class data
X0_data=sm_data[sm_target==0,:]
X0_target=sm_target[sm_target==0]
X1_data=sm_data[sm_target==1,:]
X1_target=sm_target[sm_target==1]
print('Class 0 count information after SMOTE', X0_data.shape)
print('Class 0 count information after SMOTE', X0_target.shape)
print('Class 1 count information after SMOTE', X1_data.shape)
print('Class 1 count information after SMOTE', X1_target.shape)

"""
#=======================================================
#dimensionality reduction  ...
"""
X0_data=sm_data[sm_target==0,:]
X0t=X0_data.transpose()
X1_data=sm_data[sm_target==1,:]
X1t=X1_data.transpose()

P0, D0, Q0 = np.linalg.svd(X0t, full_matrices=False)
P1, D1, Q1 = np.linalg.svd(X1t, full_matrices=False)

ndimr=10
P0r=P0[:,0:ndimr]
P1r=P1[:,0:ndimr]

X0tr=np.matmul(P0r.transpose(),X0t)
X1tr=np.matmul(P1r.transpose(),X1t)

X0r=X0tr.transpose()
X1r=X0tr.transpose()

nt0=X0r.shape[0]
nt1=X1r.shape[0]
target0=np.zeros((nt0,))
target1=np.ones((nt1,))

svd_red_target=np.concatenate((target0, target1))
svd_red_data=np.concatenate((X0r, X1r))

np.savez_compressed(fpath+'svd_red_data_targets.npz', svd_red_data,svd_red_target)
"""
#load the data ...
smote_dict=np.load(fpath+'smote_data_targets.npz')
mlearn_data=smote_dict['arr_0']
mlearn_target=smote_dict['arr_1']

svd_dict=np.load(fpath+'svd_red_data_targets.npz')
mlearn_data=svd_dict['arr_0']
mlearn_target=svd_dict['arr_1']

#==========================================================
#create training and validation sets ....
#test_size=% for validation ...
# X= input, Y= output
X_train, X_validate, Y_train, Y_validate = train_test_split(mlearn_data, mlearn_target, test_size=0.2, random_state=1)

#==================================================
#choose some classifier algorithms .....
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
#massive distance matrix --> models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#massive optimization problem --> models.append(('SVM', SVC(gamma='auto')))

#s===================================================
#summarize training results ....
# evaluate how well each model trained to the training set 
# takes about 4 minutes (if KNN and SVM left out)
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#================================================
# test models with validation set
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validate)

print(accuracy_score(Y_validate, predictions))
print(confusion_matrix(Y_validate, predictions))
print(classification_report(Y_validate, predictions))


#============================================================














