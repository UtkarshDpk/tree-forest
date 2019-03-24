#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

# We will use this classification algorithm to build a model from historical data of patients, and their response to different medications. Then we use the trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.

# Import the Following Libraries:
# <ul>
#     <li> <b>numpy (as np)</b> </li>
#     <li> <b>pandas</b> </li>
#     <li> <b>DecisionTreeClassifier</b> from <b>sklearn.tree</b> </li>
# </ul>

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ### About dataset
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. 
# 
# Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to. 
# 
# It is a sample of binary classifier, and you can use the training part of the dataset 
# to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.
# 

# ### Downloading Data
# To download the data, we will use !wget to download it from IBM Object Storage.

# In[2]:


get_ipython().system('wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')


# now, read data using pandas dataframe:

# In[2]:


my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]


# ## Pre-processing

# What is the dimensions of the dataset?

# In[13]:


print("Dimension of the dataset is\n{}".format(my_data.ndim))


# Using <b>my_data</b> as the Drug.csv data read by pandas, declare the following variables: <br>
# 
# <ul>
#     <li> <b> X </b> as the <b> Feature Matrix </b> (data of my_data) </li>
#     <li> <b> y </b> as the <b> response vector (target) </b> </li>
# </ul>

# Remove the column containing the target name since it doesn't contain numeric values.

# In[3]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# As you may figure out, some features in this dataset are categorical such as __Sex__ or __BP__. Unfortunately, Sklearn Decision Trees do not handle categorical variables. But still we can convert these features to numerical values. __pandas.get_dummies()__
# Convert categorical variable into dummy/indicator variables.

# In[4]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# Now we can fill the target variable.

# In[5]:


y = my_data["Drug"]
y[0:5]


# ---
# ## Setting up the Decision Tree
# We will be using <b>train/test split</b> on our <b>decision tree</b>. Let's import <b>train_test_split</b> from <b>sklearn.cross_validation</b>.

# In[6]:


from sklearn.model_selection import train_test_split


# Now <b> train_test_split </b> will return 4 different parameters. We will name them:<br>
# X_trainset, X_testset, y_trainset, y_testset <br> <br>
# The <b> train_test_split </b> will need the parameters: <br>
# X, y, test_size=0.3, and random_state=3. <br> <br>
# The <b>X</b> and <b>y</b> are the arrays required before the split, the <b>test_size</b> represents the ratio of the testing dataset, and the <b>random_state</b> ensures that we obtain the same splits.

# In[8]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# ## Modeling
# We will first create an instance of the <b>DecisionTreeClassifier</b> called <b>drugTree</b>.<br>
# Inside of the classifier, specify <i> criterion="entropy" </i> so we can see the information gain of each node.

# In[9]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# Next, we will fit the data with the training feature matrix <b> X_trainset </b> and training  response vector <b> y_trainset </b>

# In[10]:


drugTree.fit(X_trainset,y_trainset)


# ## Prediction
# Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.

# In[11]:


predTree = drugTree.predict(X_testset)


# You can print out <b>predTree</b> and <b>y_testset</b> if you want to visually compare the prediction to the actual values.

# In[12]:


print (predTree [0:5])
print (y_testset [0:5])


# ## Evaluation
# Next, let's import __metrics__ from sklearn and check the accuracy of our model.

# In[13]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# __Accuracy classification score__ computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  
# 
# In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# ## Visualization
# Lets visualize the tree

# In[ ]:


# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y


# In[15]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:




