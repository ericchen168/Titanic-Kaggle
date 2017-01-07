import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score



# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df.Age.fillna(titanic_df.Age.median(),inplace=True)
#titanic_df.fillna(-1,inplace=True)
titanic_df['Title'] = titanic_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
# a map of more aggregated titles
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
titanic_df['Title'] = titanic_df.Title.map(Title_Dictionary)
test_df['Title'] = test_df.Title.map(Title_Dictionary)

#test set cleaning
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
test_df.Age.fillna(test_df.Age.median(),inplace=True)
#test_df.fillna(-1,inplace=True)
titanic_df['family']=titanic_df.SibSp+titanic_df.Parch
test_df['family']=test_df.SibSp+test_df.Parch

feature=['Pclass','Sex','Age','SibSp','Parch','Fare','family','Title']

X=titanic_df[feature]
X_test=test_df[feature]

X_total=pd.concat([X,X_test])
grouped = X_total.groupby(['Sex','Pclass'])
grouped.median()

X=pd.get_dummies(X,columns=['Pclass','Sex','Title'],drop_first=True)
X_test=pd.get_dummies(X_test,columns=['Pclass','Sex','Title'],drop_first=True)
#X=pd.get_dummies(X,columns=['Pclass','Sex','Title'],drop_first=True)
#X_test=pd.get_dummies(X_test,columns=['Pclass','Sex','Title'],drop_first=True)
#
#

y=titanic_df.Survived





#X.Fare=X.Fare/ X.Fare.max()
#X_test.Fare=X_test.Fare/ X_test.Fare.max()

from sklearn import preprocessing
normalizer = preprocessing.Normalizer()

#X = preprocessing.Normalizer().fit_transform(X)
X = preprocessing.MaxAbsScaler().fit_transform(X)
X_test = preprocessing.MaxAbsScaler().fit_transform(X_test)
#X = preprocessing.MinMaxScaler().fit_transform(X)
#pd.DataFrame(preprocessing.Normalizer().fit_transform(X.Fare))

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

## KNN check
#score_knn = []
#for k in range (1,31):
#    from sklearn.cross_validation import cross_val_score
#    knn = KNeighborsClassifier(n_neighbors = k)
#    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
#    score_knn.append(scores.mean())
#
#
## KNN with grid search CV
#from sklearn.grid_search import GridSearchCV
#k_range=range(1,31)
#weight_options=['uniform','distance']
#param_grid=dict(n_neighbors=k_range,weights=weight_options)
#knn = KNeighborsClassifier()
#grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
#grid.fit(X,y)
#grid.grid_scores_
#
#grid_mean_scores=[result.mean_validation_score for result in grid.grid_scores_]
##plt.plot(k_range,grid_mean_scores)
#
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_
#
#
## SVC
#from sklearn.cross_validation import cross_val_score
#model= SVC(kernel='rbf')
#score_svc=cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()
#print score_svc
##SVC with grid CV search
#
#kernel_opt=[ 'linear', 'rbf', 'sigmoid', 'precomputed']
##C_opt=range(1, 4, 1)
#param_grid=dict(kernel=kernel_opt)#,C=C_opt)
#model= SVC()
#grid=GridSearchCV(model,param_grid,cv=5,scoring='accuracy')
#grid.fit(X,y)
#grid.grid_scores_
#
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_
#
## logistic regression check
#logreg = LogisticRegression(penalty='l1')
#score_log=cross_val_score(logreg,X,y,cv=5,scoring='accuracy').mean()
#print score_log
## Decision Tree with  gridCV
#
#
#from sklearn import tree
#from sklearn.cross_validation import cross_val_score
#
#depth_opt=range(1,15)
#criterion_opt=['entropy','gini']
#param_grid=dict(max_depth=depth_opt,criterion=criterion_opt)
#DCT = tree.DecisionTreeClassifier()
#
#grid=GridSearchCV(DCT,param_grid,cv=10,scoring='accuracy')
#grid.fit(X,y)
#grid.grid_scores_
#
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_
#
#
##Random Forest
#
#randomf = RandomForestClassifier(n_estimators=10, oob_score=True)
#score_random=cross_val_score(randomf,X,y,cv=10,scoring='accuracy').mean()
#print score_random
#
#
##Random Forest with Grid CV
from sklearn.grid_search import GridSearchCV
n_estimators_opt=range(10,35,5)
oob_score_opt=['True','False']
criterion_opt=['gini','entropy']
param_grid=dict(n_estimators=n_estimators_opt,criterion=criterion_opt,oob_score=oob_score_opt)
randomf = RandomForestClassifier()
grid=GridSearchCV(randomf,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
grid.grid_scores_

print grid.best_score_
print grid.best_params_
print grid.best_estimator_

#
##ensemble Random Forest with Grid CV
#from sklearn.grid_search import GridSearchCV
#n_estimators_opt=range(10,110,10)
#oob_score_opt=['True','False']
#criterion_opt=['gini','entropy']
#param_grid=dict(n_estimators=n_estimators_opt,criterion=criterion_opt,oob_score=oob_score_opt)
#randomf = ensemble.RandomForestClassifier()
#grid=GridSearchCV(randomf,param_grid,cv=10,scoring='accuracy')
#grid.fit(X,y)
#grid.grid_scores_
#
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_
#Gradient boost

from sklearn.ensemble import GradientBoostingClassifier
n_estimators_opt=range(20,60,5)
depth_opt=range(3,6)
param_grid=dict(n_estimators=n_estimators_opt,max_depth=depth_opt)
clfgrad = GradientBoostingClassifier(random_state=42)
grid=GridSearchCV(clfgrad,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
grid.grid_scores_


print grid.best_score_
print grid.best_params_
print grid.best_estimator_

#adaboost

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
n_estimators_opt=range(50,150,10)
param_grid=dict(n_estimators=n_estimators_opt)
clfada = AdaBoostClassifier(random_state=42)
grid=GridSearchCV(clfada,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
grid.grid_scores_


print grid.best_score_
print grid.best_params_
print grid.best_estimator_

# submit with decision tree

DCT = tree.DecisionTreeClassifier(criterion='entropy', max_depth=11)
score_dct=cross_val_score(DCT,X,y,cv=10,scoring='accuracy').mean()
DCT.fit(X,y)
prediction=DCT.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('titanic_DCT.csv', index=False)

# submit with random forest

randomf = RandomForestClassifier(n_estimators= 20,random_state=42, criterion='gini', min_samples_split=2, oob_score=True)
score_random=cross_val_score(randomf,X,y,cv=10,scoring='accuracy').mean()
randomf.fit(X,y)
prediction=randomf.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('titanic_random.csv', index=False)


# submit with adaboost
clfada = AdaBoostClassifier(n_estimators=100)
score_random=cross_val_score(clfada,X,y,cv=10,scoring='accuracy').mean()
clfada.fit(X,y)
prediction=clfada.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('titanic_random.csv', index=False)


# submit with Gradient boost
clfgrad = GradientBoostingClassifier(max_depth=4,n_estimators=45)
score_random=cross_val_score(clfgrad,X,y,cv=8,scoring='accuracy').mean()
clfgrad.fit(X,y)
prediction=clfgrad.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('titanic_grad.csv', index=False)

# Seaborn practice

#sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)
#fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
#sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
#sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)
#embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
#sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
#
#
#plt.hist(titanic_df[titanic_df.Survived==1].Fare,label='survived')
#plt.hist(titanic_df[titanic_df.Survived==0].Fare,label='not survived')
#plt.show()
#
#plt.figure()
#sns.pairplot(titanic_df,hue='Survived')
#plt.savefig("1_seaborn_pair_plot.png")