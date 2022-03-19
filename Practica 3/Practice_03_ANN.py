
# coding: utf-8

# _Python_ package [neurolab](https://github.com/zueve/neurolab) provides a working environment for ANN.

# # Perceptrons

# We can implement a neural network having a single layer of perceptrons (apart from input units) using _neurolab_ package as an instance of the class `newp`. In order to do so we need to provide the following parameters:
# * `minmax`: a list with the same length as the number of input neurons. The $i$-th element on this list is a list of two numbers, indicating the range of input values for the $i$-th neuron.
# * `cn`: number of output neurons.
# * `transf`: activation function (default value is threshold).
# 
# Therefore, when we choose 1 as the value of parameter `cn`, we will be representing a simple perceptron having as many inputs as the length of the list associated to `minmax`.

# Let us start by creating a simple perceptron with two inputs, both of them ranging in $[0, 1]$, and with threshold activation function.

# In[ ]:


from neurolab import net

perceptron = net.newp(minmax=[[0, 1], [0, 1]], cn=1)


# The instance that we just created has the following attributes:
# * `inp_minmax`: range of input values.
# * `co`: number of output neurons.
# * `trainf`: training function (the only one specific for single-layer perceptrons is the Delta rule).
# * `errorf`: error function (default value is half of SSE, _sum of squared errors_)
# 
# The layers of the neural network (input layer does not count, thus in our example there is only one) are stored in a list associated with the attribute `layers`. Each layer is an instance of the class `Layer` and has the following attributes:
# * `ci`: number of inputs.
# * `cn`: number of neurons on it.
# * `co`: number of outputs.
# * `np`: dictionary with an element `'b'` that stores an array with the neurons' biasses (terms $a_0 w_0$, default value is 0) and an element `'w'` that stores an array with the weights associated with the incoming connections arriving on each neuron (default value is 0).

# In[ ]:


print(perceptron.inp_minmax)
print(perceptron.co)
print(perceptron.trainf)
print(perceptron.errorf)
layer = perceptron.layers[0]
print(layer.ci)
print(layer.cn)
print(layer.co)
print(layer.np)


# Next, let us train the perceptron so that it models the logic gate _and_.
# 
# First of all, let us define the training set. We shall do it indicating on one hand an array or list of lists with the imput values corresponding to the examples, and on the other hand a different array or list of lists with the expected ouput for each example.

# In[ ]:


import numpy

input_values = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outcomes = numpy.array([[0], [0], [0], [1]])


# The method `step` allows us to calculate the output of the neural network for a single example, and the method `sim` for all the examples.

# In[ ]:


perceptron.step([1, 1])


# In[ ]:


perceptron.sim(input_values)


# Let us check which is the initial error of the perceptron, before the training.
# 
# __Important__: the arguments of the error function must be arrays.

# In[ ]:


perceptron.errorf(expected_outcomes, perceptron.sim(input_values))


# Let us next proceed to train the perceptron. We shall check that, as expected (since the training set is linearly separable), we are able to decrease the value of the error down to zero.
# 
# __Note__: the method `train` that runs the training algorithm on the neural network returns a list showing the value of the network error after each of the _epochs_. More precisely, an epoch represents the set of operations performed by the training algorithm until all the examples  of the training set have been considered.

# In[ ]:


perceptron.train(input_values, expected_outcomes)


# In[ ]:


print(perceptron.layers[0].np)
print(perceptron.errorf(expected_outcomes, perceptron.sim(input_values)))


# # Feed forward perceptrons

# Package _neurolab_ implements a feed forward artificial neural network as an instance of the class `newff`. In order to do so, we need to provide the following parameters:
# * `minmax`: a list with the same length as the number of input neurons. The $i$-th element on this list is a list of two numbers, indicating the range of input values for the $i$-th neuron.
# * `cn`: number of output neurons.
# * `transf`: activation function (default value is threshold).
# 
# * `size`: a list with the same length as the number of layers (except the input layer). The $i$-th element on this list is a number, indicating the number of neurons for the $i$-th layer.
# * `transf`: a list with the same length as the number of layers (except the input layer). The $i$-th element on this list is the activation function (default value is [hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_functions) for the neurons of the $i$-th layer.

# Next, let us create a neural network with two inputs ranging over $[0, 1]$, one hidden layer having two neurons and an output layer with only one neuron. All neurons should have the sigmoid function as activation function (you may look for further available activation functions at https://pythonhosted.org/neurolab/lib.html#module-neurolab.trans).

# In[ ]:


from neurolab import trans

sigmoid_act_fun = trans.LogSig()
my_net = net.newff(minmax=[[0, 1], [0, 1]], size=[2, 1], transf=[sigmoid_act_fun]*2)


# La instancia creada tiene los siguientes atributos:
# * `inp_minmax`: rangos de valores de entrada.
# * `co`: número de neuronas de salida.
# * `trainf`: función de entrenamiento, por defecto el [algoritmo de Broyden–Fletcher–Goldfarb–Shanno](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).
# * `errorf`: función de error, por defecto la mitad de la suma de los errores al cuadrado (SSE, _sum of squared errors_)
# 
# Las capas de la red neuronal (distintas de la capa de entrada) se guardan como una lista en el atributo `layers`. Cada capa es una instancia de la clase `Layer` y tiene los siguientes atributos:
# * `ci`: número de entradas.
# * `cn`: número de neuronas.
# * `co`: número de salidas.
# * `np`: diccionario con un elemento `'b'` que guarda un array con los sesgos (los términos $a_0 w_0$) de las neuronas y un elemento `'w'` que guarda un array con los pesos de las conexiones que llegan a cada neurona. Los valores iniciales de los sesgos y de los pesos se calculan por defecto con un algoritmo específico llamado [algoritmo de inicialización de Nguyen-Widrow](https://web.stanford.edu/class/ee373b/nninitialization.pdf).

# In[ ]:


print(my_net.inp_minmax)
print(my_net.co)
print(my_net.trainf)
print(my_net.errorf)
hidden_layer = my_net.layers[0]
print(hidden_layer.ci)
print(hidden_layer.cn)
print(hidden_layer.co)
print(hidden_layer.np)
output_layer = my_net.layers[1]
print(output_layer.ci)
print(output_layer.cn)
print(output_layer.co)
print(output_layer.np)


# It is possible to modify the initialization of the biases and weights, you may find available initialization options at https://pythonhosted.org/neurolab/lib.html#module-neurolab.init.<br>
# Let us for example set all of them to zero, using the following instructions:

# In[ ]:


from neurolab import init

for l in my_net.layers:
    l.initf = init.init_zeros
my_net.init()
print(hidden_layer.np)
print(output_layer.np)


# It is also possible to modify the training algorithm, you may find available implemented options at https://pythonhosted.org/neurolab/lib.html#module-neurolab.train.<br>
# Let us for example switch to the _gradient descent backpropagation_, using the following instructions:

# In[ ]:


from neurolab import train

my_net.trainf = train.train_gd


# Finally, we can also modify the error function to be used when training, you may find available options at https://pythonhosted.org/neurolab/lib.html#module-neurolab.error.<br>
# Let us for example choose the _mean squared error_, using the following instructions:

# In[ ]:


from neurolab import error

my_net.errorf = error.MSE()


# Next, let us train our neural network so that it models the behaviour of the _xor_ logic gate.
# 
# First, we need to split our training set into two components: on one hand an array or a list of lists with the input data corresponding to each example, *xor_in* , and on the other hand an array or list of lists with the correct expected ouput for each example, *xor_out* (remember that this time the training set is **not** linearly separable).

# In[ ]:


xor_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_out = numpy.array([[0], [1], [1], [0]])


# Let us measure which is the error associated to the initial neural network before the training starts:

# In[ ]:


print(my_net.sim(xor_in))
print(my_net.errorf(xor_out, my_net.sim(xor_in)))


# Let us now proceed to run the training process on the neural network. The functions involved in the training work over the following arguments:
# * `lr`: _learning rate_, default value 0.01.
# * `epochs`: maximum number of epochs, default value 500.
# * `show`: number of epochs that should be executed between two messages in the output log, default value 100.
# * `goal`: maximum error accepted (halting criterion), default value 0.01.

# In[ ]:


my_net.train(xor_in, xor_out, lr=0.1, epochs=50, show=10, goal=0.001)
my_net.sim(xor_in)


# Let us now try a different setting. If we reset the neural network and we choose random numbers as initial values for the weights, we obtain the following:

# In[ ]:


numpy.random.seed(3287426346)  # we set this init seed only for class, so that we always get
                               # the same random numbers and we can compare
my_net.reset()
for l in my_net.layers:
    l.initf = init.InitRand([-1, 1], 'bw')  # 'b' means biases will be modified,
                                            # and 'w' the weights
my_net.init()
my_net.train(xor_in, xor_out, lr=0.1, epochs=10000, show=1000, goal=0.001)
my_net.sim(xor_in)


# # _Iris_ dataset

# _Iris_ is a classic multivariant dataset that has been exhaustively studied and has become a standard reference when analysing the behaviour of different machine learning algorithms.
# 
# _Iris_ gathers four measurements (length and width of sepal and petal) of 50 flowers of each one of the following three species of lilies: _Iris setosa_, _Iris virginica_ and _Iris versicolor_.
# 
# Let us start by reading the data from the file `iris.csv` that has been provided together with the practice. It suffices to evaluate the following expressions:

# In[ ]:


import pandas

iris = pandas.read_csv('iris.csv', header=None,
                       names=['Sepal length', 'sepal width',
                              'petal length', 'petal width',
                              'Species'])
iris.head(10)  # Display ten first examples


# Next, let us move to use a numerical version of the species instead.<br>
# Then, we should distribute the examples into two groups: training and test, and split each group into two components: input and expected output (goal).

# In[ ]:


#import pandas

iris2 = pandas.read_csv('iris_enc.csv', header=None,
                       names=['Sepal length', 'sepal width',
                              'petal length', 'petal width',
                              'Species'])
iris2.head(5)  # Display ten first examples


# In[ ]:


#this piece of code might cause an error if wrong version of sklearn
#from sklearn import preprocessing
#from sklearn import model_selection

#iris_training, iris_test = model_selection.train_test_split(
#    iris, test_size=.33, random_state=2346523,
#    stratify=iris['Species'])

#ohe = preprocessing.OneHotEncoder(sparse = False)
#input_training = iris_training.iloc[:, :4]
#goal_training = ohe.fit_transform(iris_training['Species'].values.reshape(-1, 1))
#input_training = iris_test.iloc[:, :4]
#goal_training = ohe.transform(iris_test['Species'].values.reshape(-1,1))


#################
#try this instead if the previous does not work
import pandas
from sklearn import preprocessing
from sklearn import model_selection

iris2 = pandas.read_csv('iris_enc.csv', header=None,
                       names=['Sepal length', 'sepal width',
                              'petal length', 'petal width',
                              'Species'])
#iris2.head(10)  # Display ten first examples
iris_training, iris_test = model_selection.train_test_split(
    iris2, test_size=.33, random_state=2346523,
    stratify=iris['Species'])

ohe = preprocessing.OneHotEncoder(sparse = False)

input_training = iris_training.iloc[:, :4]
goal_training = ohe.fit_transform(iris_training['Species'].values.reshape(-1, 1))
goal_training[:10]  # this displays the 10 first expected output vectors (goal)
                    # associated with the training set examples


# In[ ]:


input_test = iris_test.iloc[:, :4]
goal_test = ohe.transform(iris_test['Species'].values.reshape(-1,1))


# In[ ]:


print(input_training.head(10))
print(goal_training[:10])


# In[ ]:


print(input_test.head(10))
print(goal_test[0:10])


# __Exercise 1__: define a function **lily_species** that, given an array with three numbers as input, returns the position where the maximum value is.
# In[]:


# __Exercise 2__: Create a feed forward neural network having the following features:
# 1. Has four input neurons, one for each attribute of the iris dataset.
# 2. Has three output neurons, one for each species.
# 3. Has one hidden layer with two neurons.
# 4. All neurons of all layers use the sigmoid as activation function.
# 5. The initial biases and weights are all equal to zero.
# 6. Training method is gradient descent backpropagation.
# 7. The error function is the mean squared error.
# 
# Once you have created it, train the network over the sets `input_training` and `goal_training`.
#In[]:
netEx2 = net.newff(minmax=[[4.0, 8.5], [1.5, 5.0], [0.5, 7.5], [0.0, 3.0]], size=[2,3], transf=[sigmoid_act_fun, sigmoid_act_fun]])
for l in netEx2.layers:
    l.initf = init.init_zeros
netEx2.init()

netEx2.trainf = train.train_gd

netEx2.errorf = error.MSE()


# __Exercise 3__: Calculate the performance of the network that was trained on the previous exercise, using to this aim the sets `input_test` and `goal_test`. That is, calculate which fraction of the test set is getting the correct classification predicted by the network.<br>
# __Hint:__ In order to translate the output of the network and obtain which is the species predicted, use the function from exercise 1.

# __Exercise 4__: try to create different variants of the network from exercise 2, by modifying the number of hidden layers and/or the amount of neurons per layer, in such a way that the performance over the test set is improved.

# %%
