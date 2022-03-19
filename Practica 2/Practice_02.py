
# coding: utf-8

# In this practice we shall use _Python_'s library [scikit-learn](http://scikit-learn.org) (noted as _sklearn_ in what follows) that provides various useful tools related to machine learning.

# # Supervised learning

# In order to illustrate the concept of supervised learning we shall make use of the data set [_Car Evaluation_](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation) from [UCI](http://archive.ics.uci.edu/ml/) Machine Learning Repository. This data set includes information about cars acceptability, according to the values of the following input attributes:
# * _buying_: buying price. Possible values: `vhigh`, `high`, `med`, `low`.
# * _maint_: price of the maintenance. Possible values: `vhigh`, `high`, `med`, `low`.
# * _doors_: number of doors. Possible values: `2`, `3`, `4`, `5more`.
# * _persons_: capacity in terms of persons to carry. Possible values: `2`, `4`, `more`.
# * _lug\_boot_: the size of luggage boot. Possible values: `small`, `med`, `big`.
# * _safety_: estimated safety of the car. Possible values: `low`, `med`, `high`.
# 
# The target concept to be learnt is _acceptability_, featuring the following class values: `unacc`, `acc`, `good` or `vgood`.

# Run the code in the cell below to read the data from the file `cars.csv` that comes along with this practice. Note that [_Pandas_](http://pandas.pydata.org/) and [_NumPy_](http://www.numpy.org/) are _Python_ libraries for data analysis and scientific computing, respectively.

# In[ ]:


import pandas
import numpy

cars = pandas.read_csv('cars.csv', header=None,
                       names=['buying', 'maint', 'doors', 'persons',
                              'lug_boot', 'safety', 'acceptability'])
print(cars.shape)  # Number of rows and columns
cars.head(10)  # display 10 first rows


# _sklearn_ cannot work directly over the previous dataset, since the library assumes that the values of discrete variables are integers.
# In order to "translate" or encode the data into the right format, the library offers various preprocessing operators, e.g. `OrdinalEncoder`, that deals with input attributes, and `LabelEncoder`, that deals with the goal variable.

# In[ ]:


from sklearn import preprocessing

attributes = cars.loc[:, 'buying':'safety']  # select columns with attributes' values
objective = cars['acceptability']  # select the goal column

# In order to encode the data, we shall create an instace of the corresponding type of encoder
# and we use method *fit* on the available data.
# Methods *transform* (resp. *inverse_transform*) allow us to encode (resp. decode) data.


# In[ ]:


# In this example, the right choice is OrdinalEncoder, working over the whole array with attributes values
encoder_attr = preprocessing.OrdinalEncoder()
encoder_attr.fit(attributes)
print(encoder_attr.categories_)  # Categories detected by the encoder for each attribute
encoded_attributes = encoder_attr.transform(attributes)
print(encoded_attributes)
print(encoder_attr.inverse_transform([[3., 3., 0., 0., 2., 1.],
                                               [1., 1., 3., 2., 0., 1.]]))


# In[ ]:


# Now, the right choice for the goal variable is LabelEncoder, working on the list (not an array) of values.
encoder_goal = preprocessing.LabelEncoder()
encoded_goal = encoder_goal.fit_transform(objective)  # The *fit_transform* method adjusts the encoding
#and applies it to the data
print(encoder_goal.classes_)  # Classes detected by the encoder for the goal variable
print(encoded_goal)
print(encoder_goal.inverse_transform([2, 1, 3]))


# Once the variables have been encoded, we need to split the data set into two parts: a training set, that will be used to generate the models; and a test set, that will be used to compare them.
# 
# It is important to notice that the distribution of the examples among the different acceptability outcomes is not uniform: there are 1210 cars (70.023 %) classified as unacceptable (`unacc`), 384 cars (22.222 %) classified as acceptable (`acc`), 69 cars (3.993 %) classified as good (`good`) and 65 cars (3.762 %) classified as very good (`vgood`).
# 
# In order to split the data set into training and test subsets fulfilling the stratification condition, _sklearn_ provides the method `train_test_split`.

# In[ ]:


from sklearn import model_selection

print(cars.shape[0])  # Total number of examples
print(pandas.Series(objective).value_counts(normalize=True))  
# Frequency of each acceptability class

attr_train, attr_test, objective_train, objective_test = model_selection.train_test_split(
    encoded_attributes, encoded_goal,  # Data sets to be splitted, using
                                       # the same indices for both of them
    random_state=12345,  # This makes it possible to reproduce the same random sampling
    test_size=.33,  
    stratify=encoded_goal)  # Stratification wrt distribution of values in the goal variable

# Let us check that the test set indeed contains 33 % of data, and the stratification condition
print(attr_test.shape[0],
      len(objective_test),
      1728 * .33)
print(pandas.Series(encoder_goal.inverse_transform(objective_test)).value_counts(normalize=True))

# Let us check that the train set contains the rest of data, and the stratification condition
print(attr_train.shape[0],
      len(objective_train),
      1728 * .67)
print(pandas.Series(objective_train).value_counts(normalize=True))


# Let us now proceed to run examples of supervised learning on _sklearn_. It suffices to create an instance of the corresponding objects class implementing the desired model (decision trees, _naive_ Bayes, _kNN_, etc.).
# 
# Each of these instances will feature the following methods:
# * Method `fit` runs the model training, given __separately__ the training set and the value of the goal variable for each of the examples in the training set.
# * Method `predict` is used to classify a new example once the model has been trained.
# * Method `score` calculates the performance of the model, given __separately__ the test set and the value of the goal variable for each of the examples in the test set.

# ### Decision trees

# _sklearn_ implements decision trees used as classifier tools as instances of the class `DecisionTreeClassifier`.
# 
# Unfortunately, they are binary decision trees built using an algorithm different from _ID3_, assuming continuous numerical attributes, but scikit-learn implementation does not support categorical variables for now.
# 
# Please visit http://scikit-learn.org/stable/modules/tree.html for information about decision trees implemented in _sklearn_.

# ### kNN

# _sklearn_ implements _kNN_ by using instances of the class `KNeighborsClassifier`. Please refer to http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html for a description of the distances currently available.

# In[ ]:


from sklearn import neighbors

clasif_kNN = neighbors.KNeighborsClassifier(n_neighbors=5, metric='hamming')


# Let us train the model.

# In[ ]:


clasif_kNN.fit(attr_train, objective_train)


# Method `kneighbors` allows us to find which are (the indices of) the $k$ nearest neighbours of the provided examples, together with their associated distances.

# In[ ]:


new_examples = [['vhigh', 'vhigh', '3', 'more', 'big', 'high'],
                   ['high', 'low', '3', '2', 'med', 'med']]

distances, neighbours = clasif_kNN.kneighbors(encoder_attr.transform(new_examples))

# Nearest neighbours and their distance to the first new example
print(new_examples[0])
print(encoder_attr.inverse_transform(attr_train[neighbours[0]]))
print(distances[0])
print(encoder_goal.inverse_transform(objective_train[neighbours[0]]))

# Nearest neighbours and their distance to the second new example
print(new_examples[1])
print(encoder_attr.inverse_transform(attr_train[neighbours[1]]))
print(distances[1])
print(encoder_goal.inverse_transform(objective_train[neighbours[1]]))


# Method `predict` returns the outcome of the objective attribute that has been predicted by the model for a new example, and method `score` returns the _accuracy_ over a test set.

# In[ ]:


encoder_goal.inverse_transform(clasif_kNN.predict(encoder_attr.transform(new_examples)))


# In[ ]:


clasif_kNN.score(attr_test, objective_test)


# ## Applications for nursery schools

# The file `nursery.csv` provides a dataset about the ranking process over applications for nursery schools, according to the values of the following input attributes:
# * _parents_. Possible values: `usual`, `pretentious`, `great_pret`.
# * _has\_nurs_. Possible values: `proper`, `less_proper`, `improper`, `critical`, `very_crit`.
# * _form_. Possible values: `complete`, `completed`, `incomplete`, `foster`.
# * _children_. Possible values: `1`, `2`, `3`, `more`.
# * _housing_. Possible values: `convenient`, `less_conv`, `critical`.
# * _finance_. Possible values: `convenient`, `inconv`.
# * _social_. Possible values: `non-prob`, `slightly_prob`, `problematic`.
# * _health_. Possible values: `recommended`, `priority`, `not_recom`.
# 
# Data are taken from an expert system that was used in the 80's in Liubliana (Slovenia), and that was developed with the aim of granting a way of providing an objective explanation to the parents getting their applications rejected.
# 
# The evaluation of each application is recorded on the attribute _evaluation_, that features five different classification outcomes: `not_recom`, `recommend`, `very_recom`, `priority` or `spec_prior`.
# 
# Our goal is to learn, based on the provided data, a model able to predict in the best possible way which will be the classification outcome for a given application, according to the values of the input attributes.
# 
# To this aim, you should follow these steps:

# * Read the data from the file `nursery.csv`.

# * Encode the data using integer values.

# * Split the data set into two subsets: a training set (80 % of the examples) and a test set (20 % of the examples).
# The first subset will be used for cross validation so that we can choose the best model to be trained. The second subset will be used to measure the accuracy.

# * Using the training subset and the crossvalidation technique with 10 partitions, provide an estimation of the average _accuracy_ of a kNN-type model, for each one of the possible values k = 1, ..., 10.

# __Note__: the function `cross_val_score` of the module `model_selection` of _sklearn_ implements the crossvalidation process. Such function takes, among other options, the following arguments:
# * _estimator_: model under evaluation.
# * _X_: array with the values of the input attributes for the given examples.
# * _y_: array with the classification values of such examples.
# * _cv_: number of partitions into which the data should be divided.
# 
# Returns an array with the accuracy of the model over each one of the subsets, after training the model using the examples from all complementary subsets.
# 
# For more information about the implementation of crossvalidation method in _sklearn_ please visit http://scikit-learn.org/stable/modules/cross_validation.html.

# * Using the best $k$ value found, train the _kNN_ algorithm and compute the average accuracy over the test set.

# In[] 
import pandas
import numpy
nurses = pandas.read_csv('nursery.csv', header=None,
                       names=['parents','has_nurs','form','children','housing','finance','social','health', 'acceptability'])
print(nurses.shape)  # Number of rows and columns
nurses.head(10)

# In[] 
from sklearn import preprocessing

attributes = nurses.loc[:, 'parents':'health']  # select columns with attributes' values
objective = nurses['acceptability']  # select the goal column

# %%
encoder_attr = preprocessing.OrdinalEncoder()
encoder_attr.fit(attributes)
print(encoder_attr.categories_)  # Categories detected by the encoder for each attribute
encoded_attributes = encoder_attr.transform(attributes)
print(encoded_attributes)
print(encoder_attr.inverse_transform([[3., 3., 0., 0., 2., 1.],
                                               [1., 1., 3., 2., 0., 1.]]))

# %%
