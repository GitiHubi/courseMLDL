#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#-*- coding: utf-8 -*-


# <img align="right" style="max-width: 200px; height: auto" src="hsg_logo.png">
# 
# ###  Lab 03 - "Supervised Machine Learning"
# 
# Introduction to ML and DL, University of St. Gallen, Autumn Term 2019

# In the last lab, you learned about Python programming elements such as conditions, loops as well as how to implement functions etc. In this third lab, we will build our first supervised machine learning "pipeline" using:
# 
# 
# - (1) the **Gaussian Naive-Bayes (Gaussian NB)** classifier, and; 
# - (2) the **k Nearest-Neighbours (kNN)** classifier 
# 
# you learned about in the lecture.
# 
# The **Naive-Bayes (NB)** classifier belongs to the family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes has been studied extensively since the 1950s and remains an accessible (baseline) method for text categorization as well as other domains.
# 
# 
# The **k Nearest-Neighbours (kNN)** is a simple, easy to understand, versatile, but powerful machine learning algorithm. Until recently (prior to the advent of deep learning approaches) it was used in a variety of applications such as finance, healthcare, political science, handwriting detection, image recognition and video recognition. In Credit ratings, financial institutes will predict the credit rating of customers. 

# As always, pls. don't hesitate to ask all your questions either during the lab, post them in our CANVAS (StudyNet) forum (https://learning.unisg.ch), or send us an email (using the course email).

# ### Lab Objectives:

# After today's lab you should be able to:
# 
# > 1. Understand how a Gaussian **Naive-Bayes (NB)** classifier can be trained and evaluated.
# > 2. Understand how a **k Nearest-Neighbor (kNN)** classifier can be trained and evaluated.
# > 3. Know how to Python's sklearn library to **train** and **evaluate** arbitrary classifiers.
# > 4. Understand how to **evaluate** and **interpret** the classification results.

# Before we start let's watch a motivational video:

# In[ ]:


from IPython.display import YouTubeVideo
# GTC 2017: "I Am AI" Opening in Keynote
# YouTubeVideo('SUNPrR4o5ZA', width=1024, height=576)


# ### Step 0: Setup of the Analysis Environment

# Similar to the previous labs, we need to import a couple of Python libraries that allow for data analysis and data visualization. For the purpose of this lab will use the pandas, numpy, sklearn, matplotlib and seaborn library. Let's import the libraries by execution of the statements below:

# In[1]:


# import the numpy, scipy and pandas data science library
import pandas as pd
import numpy as np
from scipy.stats import norm

# import sklearn data and data pre-processing libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split

# import sklearn naive.bayes and k-nearest neighbor classifier library
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# import sklearn classification evaluation library
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# import matplotlib data visualization library
import matplotlib.pyplot as plt
import seaborn as sns


# Enable inline Jupyter notebook plotting:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## A. Gaussian "Naive-Bayes" (NB) Classification

# ### Step 1.0: Dataset Download and Data Assessment

# The iris dataset is a classic and very easy multi-class classification dataset. This data sets consists of 3 different types of irises’ (classes),  namely Setosa, Versicolour, and Virginica) as well as their respective petal and sepal length (features).

# <img align="center" style="max-width: 700px; height: auto" src="iris_dataset.png">
# 
# (Source: http://www.lac.inpe.br/~rafael.santos/Docs/R/CAP394/WholeStory-Iris.html)

# In total, the dataset consists of **150 samples** (50 samples per class) as well as their corresponding **4 different measurements** taken for each sample. Please, find below the list of the individual measurements (features):
# 
# >- `Sepal length (cm)`
# >- `Sepal width (cm)`
# >- `Petal length (cm)`
# >- `Petal width (cm)`
# 
# Further details on the dataset can be obtained from the following puplication: *Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950)."*
# 
# Let's load the dataset and conduct a preliminary data assessment: 

# In[2]:


iris = datasets.load_iris()


# Print and inspect feature names of the dataset:

# In[3]:


iris.feature_names


# Print and inspect the class names of the dataset:

# In[4]:


iris.target_names


# Let's briefly envision how the feature data is collected and recorded in the data:

# <img align="center" style="max-width: 900px; height: auto" src="feature_collection.png">

# Print and inspect the top 5 feature rows of the dataset:

# In[ ]:


pd.DataFrame(iris.data, columns=iris.feature_names).head(5)


# Print and inspect the top 5 labels of the dataset:

# In[ ]:


pd.DataFrame(iris.target, columns=["class"]).head(5)


# Determine and print the feature dimensionality of the dataset:

# In[ ]:


iris.data.shape


# Determine and print the label dimensionality of the dataset:

# In[ ]:


iris.target.shape


# Plot the data distributions of the distinct features and its class membership respectively:

# In[ ]:


plt.figure(figsize=(10, 10))
iris_plot = sns.load_dataset("iris")
sns.pairplot(iris_plot, diag_kind='hist', hue='species');


# ### Step 1.1. Dataset Pre-Processing

# In order to understand and evaluate the performance of any trained model, it is good practice to divide the dataset into a **training set** (the fraction of records soley used for training purposes) and a **evaluation set** (the fraction of records soley used for evaluation purposes). Pls. note, the **evaluation set** will never shown to the model as part of the training process.

# We set the fraction of testing records to **30%** of the original dataset:

# <img align="center" style="max-width: 500px; height: auto" src="train_eval_dataset.png">

# In[ ]:


eval_fraction = 0.3


# Randomly split the dataset into training set and evaluation set using sklearns `train_test_split` function:

# In[ ]:


# 70% training and 30% evaluation
x_train, x_eval, y_train, y_eval = train_test_split(iris.data, iris.target, test_size=eval_fraction)


# Evaluate the training set dimensionality:

# In[ ]:


x_train.shape, y_train.shape


# Evaluate the evaluation set dimensionality:

# In[ ]:


x_eval.shape, y_eval.shape


# ### Step 1.2. Gaussian Naive-Bayes (NB) Classification

# In probability theory and statistics, the **Bayes' theorem** (alternatively Bayes' law or Bayes' rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. 
# 
# As you learned in the lecture the Bayes' theorem provides a way of calculating posterior probability $P(c|x)$. Let's briefly revisit the Bayes equation below:

# <img align="center" style="max-width: 400px; height: auto" src="bayes_theorem.png">

# Above,
# 
# >- $P(c|x)$ is the **posterior** probability of class (c, target) given a predictor (x, attributes).
# >- $P(c)$ is the **prior** probability of a class.
# >- $P(x|c)$ is the **likelihood** which is the probability of a predictors given class.
# >- $P(x)$ is the **evidence** of a feature also sometimes referred to as predictor.

# #### Step 1.2.1. Calculation of the prior probabilities $P(c)$ of each class

# Let's get an intuition of the Bayes' theorem by calculating the prior probability $P(c)$ of each class. Therefore, we first obtain the number of occurance of each class in the training data:

# In[ ]:


unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
print(class_counts)  


# Let's convert the obtained counts into probabilites by dividing the class counts by the overall number of observations:

# In[ ]:


prior_probabilities = counts / x_train.shape[0]
print(prior_probabilities)


# #### Step 1.2.2. Calculation of the evidence $P(x)$ of each feature

# Let's now calculate the evidence $P(x)$ of each feature. During the lecture we learned that we can approximate $P(x)$ by a Gaussian (Normal) probability distribution $\mathcal{N}(\mu, \sigma)$ applying the "law of large numbers" or "Central Limit Theorem" (you may want to have a look at further details of the theorem under: https://en.wikipedia.org/wiki/Central_limit_theorem). 
# 
# The **evidence** probability density of a Gaussian "Normal" distribution, as defined by the formula below, is determined by its mean $\mu$ and standard deviation $\sigma$:

# <img align="center" style="max-width: 600px; height: auto" src="evidence_calculation.png">

# We will approximate the probability density $P(x) \approx \mathcal{N}(x | \mu, \sigma)$ of each of each feature by a Gaussian. But how can this be achieved? 
# 
# Let's start by inspecting the true probability density of the **sepal length** feature (the first feature) of the iris dataset. The following line of code determines a histogram of the true feature value distribution:

# In[ ]:


# determine a histogram of the feature value distribution
hist, bin_edges = np.histogram(x_train[:, 0], bins=10, density=True)
print(hist)
print(bin_edges)


# Let's also plot the probability density accordingly:

# In[ ]:


# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# plot histogram of "sepal length" observations
ax.hist(x_train[:, 0], bins=10, density=True, color='g')

# add grid
ax.grid(linestyle='dotted')

# add axis range and legends
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$P(x)$", fontsize=10)
ax.set_ylim([0.0, 0.6])

# add plot title
ax.set_title('Sepal Length', fontsize=10);


# How can we approximate the true probability density of the **sepal length** feature? Well we need calculate it's mean $\mu$ and standard deviation $\sigma$. Let's start by calculating the mean $\mu$ of the **sepal length** feature:

# In[ ]:


mean = np.mean(x_train[:, 0])
print(mean)


# Let's continue by calculating the standard devition $\sigma$ of the **sepal length** feature:

# In[ ]:


std = np.std(x_train[:, 0])
print(std)


# We can now determine the approximate Gaussian (Normal) probability density distribution $\mathcal{N}(\mu, \sigma)$ of the **sepal length** feature using the $\mu$ and $\sigma$ obtained above as well as the `pdf.norm` function of the `scipy.stats` package:

# In[ ]:


hist_gauss = norm.pdf(bin_edges, mean, std)
print(hist_gauss)


# Let's now plot the approximate Gaussian (Normal) probability density distribution $P(x) \approx \mathcal{N}(\mu, \sigma)$:

# In[ ]:


# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# plot fitted "gaussian" or normal distribution
ax.plot(bin_edges, norm.pdf(bin_edges, mean, std), color='r', linestyle='--', linewidth=2)

# add grid
ax.grid(linestyle='dotted')

# add axis range and legends
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$P(x)$", fontsize=10)
ax.set_ylim([0.0, 0.6])

# add plot title
ax.set_title('Gaussian Approximation Sepal Length', fontsize=10);


# #### Step 1.2.3. Calculation of the likelihood $P(x|c)$ of each feature

# Let's now see how we can calculate the **likelihood** $P(x|c)$ which is the probability density of a feature given a certain class $c$. We will again can estimate $P(x|c)$ by a Gaussian (Normal) probability distribution $\mathcal{N}(\mu, \sigma)$ applying the "law of large numbers".
# 
# The **likelihood** probability density of a Gaussian "Normal" distribution, as defined by the formula below, is determined by its mean $\mu$, standard deviation $\sigma$ and it's corresponding class condition $c$:

# <img align="center" style="max-width: 600px; height: auto" src="likelihood_calculation.png">

# Let's start by applying the class conditioning. This is usually done by filtering the dataset for each class $c$:

# In[ ]:


x_train_setosa = x_train[y_train == 0]
x_train_versicolor = x_train[y_train == 1]
x_train_virginica = x_train[y_train == 2]


# Let's start by inspecting the true probability density of the **sepal length** feature (the first feature) of the iris dataset given the class **setosa**. The following line of code determines a histogram of the true feature value distribution:

# In[ ]:


# determine a histogram of the feature value distribution
hist_setosa, bin_edges_setosa = np.histogram(x_train_setosa[:, 0], bins=10, range=(np.min(x_train[:, 0]), np.max(x_train[:, 0])), density=True)
print(hist_setosa)
print(bin_edges_setosa)


# Let's also plot the probability density accordingly:

# In[ ]:


# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# plot histogram of "sepal length" observations
ax.hist(x_train_setosa[:, 0], bins=10, range=(np.min(x_train[:, 0]), np.max(x_train[:, 0])), density=True, color='g')

# add grid
ax.grid(linestyle='dotted')

# add axis range and legends
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$P(x)$", fontsize=10)
ax.set_ylim([0.0, 1.5])

# add plot title
ax.set_title('Setosa Sepal Length', fontsize=10);


# We are again able to determine the approximate Gaussian (Normal) probability density distribution $\mathcal{N}(\mu, \sigma, c)$ of the **sepal length** feature given the class **setosa** using the $\mu$ and $\sigma$ obtained above as well as the `pdf.norm` function of the `scipy.stats` package.
# 
# Let's continue by calculating the mean $\mu$ of the **sepal length** feature given the class **setosa**:

# In[ ]:


mean_setosa = np.mean(x_train_setosa[:, 0])
print(mean_setosa)


# Let's continue by calculating the standard devition $\sigma$ of the **sepal length** feature given the class **setosa**:

# In[ ]:


std_setosa = np.std(x_train_setosa[:, 0])
print(std_setosa)


# In[ ]:


hist_gauss = norm.pdf(bin_edges_setosa, mean_setosa, std_setosa)
print(hist_gauss)


# Let's now plot the approximate Gaussian (Normal) probability density distribution $P(x | c) \approx \mathcal{N}(\mu, \sigma, c)$:

# In[ ]:


# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# plot fitted "gaussian" or normal distribution
ax.plot(bin_edges, norm.pdf(bin_edges_setosa, mean_setosa, std_setosa), color='r', linestyle='--', linewidth=2)

# add grid
ax.grid(linestyle='dotted')

# add axis range and legends
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$P(x)$", fontsize=10)
ax.set_ylim([0.0, 1.5])

# add plot title
ax.set_title('Gaussian Approximation Setosa Sepal Length', fontsize=10);


# Now we have determined all the necessary distributions $P(c)$, $P(x)$ and $P(x|c)$ given the class **setosa** and the **sepal length** feature in order to determine if a new (so far unknown) sepal length observation belongs to an Iris Setosa. Let's calculate the probability of a sepal length observation of 6.4 and its probability of beeing of class **setosa**: 

# In[ ]:


sepal_length = 6.4
probability = (0.3333 * norm.pdf(sepal_length, mean_setosa, std_setosa)) / norm.pdf(sepal_length, mean, std)
print(probability)


# Ok, it seems to be very unlikely that this Iris flower is of class **setosa**. Let's compare this to a sepal length observation of 5.1 and determine its probability of beeing of class **setosa**:

# In[ ]:


sepal_length = 5.4
probability = (0.3333 * norm.pdf(sepal_length, mean_setosa, std_setosa)) / norm.pdf(sepal_length, mean, std)
print(probability)


# Alright, this observations seems to be significantly more probable of class **setosa**.

# The Naive-Bayes algorithm assumes that your features are independent (hence we call it "naive", since it makes the naive assumption about independence, so we don't have to care about dependencies between them). We can use the independence assumption to enhance the single feature model outlined above to multiple features by defining a dot product as defined below: 

# <img align="center" style="max-width: 800px; height: auto" src="multiple_features.png">

# #### Step 1.2.4. Training and utilization of a Gaussian Naive-Bayes Classifier using Python's Sklearn library

# Luckily, there is Python library named `sklearn` that provides a variety of supervised classification algorithms which we will use in the following. 
# 
# Let's init the **Gaussian Naive-Bayes (GaussianNB)** classifier of `sklearn`: 

# In[ ]:


gnb = GaussianNB()


# Train or fit the GaussianNB classifier using the training dataset features and labels:

# In[ ]:


gnb.fit(x_train, y_train)


# Utilize the trained model to predict the response for the evaluation dataset:

# In[ ]:


y_pred = gnb.predict(x_eval)


# Let's have a look at the predicted class labels:

# In[ ]:


y_pred


# As well as the true class labels:

# In[ ]:


y_eval


# Determine **prediction accuracy** of the trained model on the evaluation dataset:

# In[ ]:


print("Accuracy: ", metrics.accuracy_score(y_eval, y_pred))


# Determine number of missclassified data sampels in the evaluation dataset:

# In[ ]:


print("Number of mislabeled points out of a total {} points: {}".format(x_eval.shape[0], np.sum(y_eval != y_pred)))


# In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm. Each row of the Matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).

# <img align="center" style="max-width: 300px; height: auto" src="confusion_matrix.png">
# 
# (Source: https://en.wikipedia.org/wiki/Confusion_matrix)

# Determine and plot the **confusion matrix** of the individual predictions:

# In[ ]:


mat = confusion_matrix(y_eval, y_pred)


# In[ ]:


sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='BuGn_r', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('NB Confusion Matrix')
plt.xlabel('[true label]')
plt.ylabel('[predicted label]');


# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Train and evaluate the prediction accuracy of distinct train- vs. eval-data ratios.**
# 
# > Change the ratio of training data vs. evaluation data to 30%/70% (currently 70%/30%), fit your model and calculate the new classification accuracy. Subsequently, repeat the experiment a second time using a 10%/90% fraction of training data/evaluation data. What can be observed in both experiments in terms of classification accuracy? 

# In[ ]:





# **2. Calculate the true-positive as well as false-positive rate of the Iris versicolor vs. virginica.**
# 
# > Calculate the true-positive rate as well as false-positive rate of (1) the experiment exhibiting a 30%/70% ratio of training data vs. evaluation data and (2) the experiment exhibiting a 10%/90% ratio of training data vs. evaluation data.

# In[ ]:





# ## B. k Nearest-Neighbor (k-NN) Classification

# ### Step 2.1: Dataset Download and Data Assessment

# The **"Wine"** dataset is a classic and very easy multi-class classification dataset. The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators (types). The dataset consists of in total **178 wines** as well as their corresponding **13 different measurements** taken for different constituents found in the three types of wine. Please, find below the list of the individual measurements (features):
# 
# >- `Alcohol`
# >- `Malic acid`
# >- `Ash`
# >- `Alcalinity of ash`
# >- `Magnesium`
# >- `Total phenols`
# >- `Flavanoids`
# >- `Nonflavanoid phenols`
# >- `Proanthocyanins`
# >- `Color intensity`
# >- `Hue`
# >- `OD280/OD315 of diluted wines`
# >- `CProline`
# 
# Further details on the dataset can be obtained from the following puplication: *"Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy."*
# 
# Let's load the dataset and conduct a preliminary data assessment: 

# In[ ]:


wine = datasets.load_wine()


# Print and inspect feature names of the dataset:

# In[ ]:


wine.feature_names


# Print and inspect the class names of the dataset:

# In[ ]:


wine.target_names


# Print and inspect the top 10 feature rows of the dataset:

# In[ ]:


pd.DataFrame(wine.data).head(10)


# Print and inspect the top 10 labels of the dataset:

# In[ ]:


pd.DataFrame(wine.target).head(10)


# Determine and print the feature dimensionality of the dataset:

# In[ ]:


wine.data.shape


# Determine and print the label dimensionality of the dataset:

# In[ ]:


wine.target.shape


# Plot the data distributions of the distinct features:

# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(pd.DataFrame(wine.data, columns=wine.feature_names))


# ### 2.2. Dataset Pre-Processing

# In order to understand and evaluate the performance of the trained k-NN model, it is good practice to divide the dataset into a **training set** (the fraction of records soley used for training purposes) and a **evaluation set** (the fraction of records soley used for evaluation purposes). Pls. note, the **evaluation set** will never shown to the model as part of the training process.
# 
# We set the fraction of testing records to 30% of the original dataset:

# In[ ]:


eval_fraction = 0.3


# Randomly split the dataset into training set and evaluation set using sklearns `train_test_split` function:

# In[ ]:


# 70% training and 30% evaluation
X_train, X_eval, y_train, y_eval = train_test_split(wine.data, wine.target, test_size=eval_fraction)


# Evaluate the training set dimensionality:

# In[ ]:


X_train.shape, y_train.shape


# Evaluate the evaluation set dimensionality:

# In[ ]:


X_eval.shape, y_eval.shape


# ### 2.3. k Nearest-Neighbor Classification

# Prior to running the **k Nearest-Neighbor (k-NN)** classification let's briefly revisit the distinct steps of the algorithm as discussed in the lecture:
# <img align="center" style="max-width: 600px; height: auto" src="hsg_knn.png">

# #### 3.1 Nearest Neighbors Classification, k=5

# Set the number of neighbors `k` to be considered in the classification of each sample: 

# In[ ]:


k_nearest_neighbors = 5


# Set the metric used in calculating the distances $D(x, x_i)$ between a sample $x$ and it's neighbors $x_i$. We will use the Euclidean distance that you learned about in the lecture, given by $\sqrt{(\sum^n_{i=1}((x - x_i)^2))}$:

# In[ ]:


distance_metric = 'euclidean'


# Init the **k-NN classifier** of Python's `sklearn` libary of data science algoritms: 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=k_nearest_neighbors, metric=distance_metric)


# Train the k-NN classifier using the training dataset:

# In[ ]:


knn.fit(X_train, y_train);


# Utilize the trained model to predict the response for the evaluation dataset:

# In[ ]:


y_pred = knn.predict(X_eval)


# Let's have a look at the predicted class labels:

# In[ ]:


y_pred


# As well as the true class labels:

# In[ ]:


y_eval


# Determine **prediction accuracy** of the trained model on the evaluation dataset:

# In[ ]:


print("Accuracy, k=5: ", metrics.accuracy_score(y_eval, y_pred))


# Determine and plot the **confusion matrix** of the individual predictions:

# In[ ]:


mat = confusion_matrix(y_eval, y_pred)


# In[ ]:


sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='BuGn_r', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('k-NN Confusion Matrix, k=5')
plt.xlabel('[true label]')
plt.ylabel('[predicted label]');


# Remember, that as part of the lecture you learned about several measures to evaluate the quality of a retrieval system, namely **Precision**, **Recall** and **F1-Score**. Let's briefly revisit their definition and subsequently calculate those measures based on the confusion matrix above:

# >- The **Precision**, denoted by Precision $=\frac{TP}{TP + FP}$, is the probability that a retrieved document is relevant.
# >- The **Recall**, denoted by Recall $=\frac{TP}{TP + FN}$, is the probability that a relevant document is retrieved.
# >- The **F1-Score**, denoted by F1-Score $= 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$, combines precision and recall is the harmonic mean of both measures.

# In[ ]:


print(classification_report(y_eval, y_pred))


# #### 3.2 Nearest Neighbors Classification, k=8

# Set the number of neighbors `k` to be considered in the classification of each sample: 

# In[ ]:


k_nearest_neighbors = 8


# Init the **k-NN classifier** of Python's `sklearn` libary of data science algoritms: 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=k_nearest_neighbors)


# Train the k-NN classifier using the training dataset:

# In[ ]:


knn.fit(X_train, y_train);


# Utilize the trained model to predict the response for the evaluation dataset:

# In[ ]:


y_pred = knn.predict(X_eval)


# Determine **prediction accuracy** of the trained model on the evaluation dataset:

# In[ ]:


print("Accuracy, k=8: ", metrics.accuracy_score(y_eval, y_pred))


# Determine and plot the **confusion matrix** of the individual predictions:

# In[ ]:


mat = confusion_matrix(y_eval, y_pred)


# In[ ]:


sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='BuGn_r', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('k-NN Confusion Matrix, k=8')
plt.xlabel('[true label]')
plt.ylabel('[predicted label]');


# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Train and evaluate the prediction accuracy of the k=1,...,40 Nearest Neighbor models.**
# 
# > Write a Python loop that trains and evaluates the prediction accuracy of all k-Nearest Neighbor parametrizations ranging from k=1,...,40. Collect and print the prediction accuracy of each model respectively and compare the results. What kind of behavior in terms of prediction accuracy can be observed with increasing k?

# In[ ]:





# **2. Plot the prediction accuracy of the k=1,...,40 Nearest Neighbor models.**
# 
# > Plot the prediction accuracy collected for each models above. The plot should display the distinct values of k at the x-axis and the corresponding model prediction accuracy on the y-axis.

# In[ ]:





# **3. Train, evaluate and plot the prediction accuracy of the k=1,...,124 Nearest Neighbor models.**
# 
# > Train, evaluate and plot the prediction accuracy of all k-Nearest Neighbor parametrizations ranging from k=1,...,124. Collect and print the prediction accuracy of each model respectively and compare the results. What kind of behavior in terms of prediction accuracy can be observed with increasing k?

# In[ ]:





# ### Lab Summary:

# In this third lab, a step by step introduction into (1) **Gaussian Naive-Bayes** and (2) **k Nearest-Neighbor** classification is presented. The code and exercises presented in this lab may serves as a starting point for more complex and tailored programs.

# You may want to execute the content of your lab outside of the Jupyter notebook environment, e.g. on a compute node or a server. The cell below converts the lab notebook into a standalone and executable python script. Pls. note, that to convert the notebook you need to install Python's **nbconvert** library and its extensions:

# In[ ]:


# installing the nbconvert library
get_ipython().system('pip install nbconvert')
get_ipython().system('pip install jupyter_contrib_nbextensions')


# **Note:** In order to execute the statement above and convert your lab notebook to a regular Python script you first need to install the nbconvert Python package e.g. using the pip package installer. 

# Let's now convert the Jupyter notebook into a plain Python script:

# In[ ]:


get_ipython().system('jupyter nbconvert --to script mldl_lab_03.ipynb')

