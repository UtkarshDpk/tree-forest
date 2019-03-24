#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# We will use this classifier to build a model from a historical dataset of irises to predict their classification based on a 
# featureset. Then we use the trained model to predict the class of a unknown iris. 
# To getter a better understanding of interaction of the dimensions we will perform Principle Component Analysis. 
# It can quickly indicate how easy or difficult the classification problem is. 
# This is particularly relevant for high-dimensional datasets.

# Import the Following Libraries:
# <ul>
#     <li> <b>numpy (as np)</b> </li>
#     <li> <b>pandas</b> </li>
#     <li> <b>RandomForestClassifier</b> from <b>sklearn.ensemble</b> </li>
# </ul>

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ### About dataset
# We use an inbuilt dataset called Iris. This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
# 
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

# In[5]:


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# To getter a better understanding of interaction of the dimensions plot the first three PCA dimensions

# In[3]:


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[37]:


dir(iris)


# <ul>
#     <li> <b> X </b> as the <b> Feature Matrix </b></li>
#     <li> <b> y </b> as the <b> response vector (target) </b> </li>
# </ul>

# ## DATA EXPLORATION...
# 
# Before starting pre-processing, basic descriptive statistics can help identify if scaling/normalization of data is needed, or how variance varies between features which is useful information for feature selection or at least understanding which features might be most important for classification
# 
# dataframe.info gives basic information on data integrity (data types and detection of NaN values)

# In[4]:


X = iris.data
y = iris.target
print('Shape of Feature Matrix is {}\nShape of Response Vector is {}'.format(X.shape,y.shape))


# Lets make a complete pandas Dataframe.

# In[39]:


df=pd.DataFrame(np.concatenate((X,y.reshape(y.shape[0],1)),axis=1),columns=iris.feature_names+['target'])
df.tail()


# In[40]:


df.info()


# Next I'm using dataframe.describe function. Works for both object and numeric. Means are in the same order of magnitude for all features so scaling might not be beneficial. If mean values were of different orders of magnitude, scaling could significantly improve accuracy of a classifier.

# In[42]:


df.describe()


# ---
# ## Setting up the Classifier
# We will be using <b>train/test split</b> on our <b>Random Forest</b>. Let's import <b>train_test_split</b> from <b>sklearn.cross_validation</b>.

# In[7]:


from sklearn.model_selection import train_test_split


# Now <b> train_test_split </b> will return 4 different parameters. We will name them:<br>
# X_trainset, X_testset, y_trainset, y_testset <br> <br>
# The <b> train_test_split </b> will need the parameters: <br>
# X, y, test_size=0.3, and random_state=3. <br> <br>
# The <b>X</b> and <b>y</b> are the arrays required before the split, the <b>test_size</b> represents the ratio of the testing dataset, and the <b>random_state</b> ensures that we obtain the same splits.

# In[8]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# ## Modeling
# We will first create an instance of the <b>RandomForestClassifier</b> called <b>irisTree</b>.<br>
# Inside of the classifier, specify <i> criterion="gini" </i> which is an index (a criterion to minimize the probability of misclassification). Note that in Decision Tree classifier the defult criterion is "entropy" which attempts to maximize the mutual information (by constructing a equal probability node) in the decision tree. Similar to entropy, the Gini index is maximal if the classes are perfectly mixed, for example, in a binary class:<br>
# $Gini = 1 - (p_1^2 + p_2^2) = 1-(0.5^2+0.5^2) = 0.5$

# In[9]:


irisTree = RandomForestClassifier(criterion="gini", max_depth = 4, n_estimators=10)
irisTree


# Next, we will fit the data with the training feature matrix <b> X_trainset </b> and training  response vector <b> y_trainset </b>

# In[10]:


irisTree.fit(X_trainset,y_trainset)


# ## Prediction
# Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.

# In[11]:


predTree = irisTree.predict(X_testset)


# You can print out <b>predTree</b> and <b>y_testset</b> if you want to visually compare the prediction to the actual values.

# In[12]:


print (predTree [0:5])
print (y_testset [0:5])


# ## Evaluation
# Next, let's import __metrics__ from sklearn and check the accuracy of our model.

# In[13]:


from sklearn import metrics
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# __Accuracy classification score__ computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  
# 
# In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# ## Visualization
# Lets visualize the results

# In[ ]:


# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y


# In[15]:


import seaborn as sn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_testset, y_pred=predTree)
tlist=iris.target_names.tolist()
df_cm = pd.DataFrame(cm, index=tlist, columns=tlist)
df_cm


# In[19]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
sn.heatmap(df_cm, annot=True)
ax.set_title('Confusion Matrix of Different Types of Irises')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')


# In[ ]:




