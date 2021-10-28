
"""
ADVANCED MACHINE LEARNING FALL 2019

@author: SARTHAK
"""

#784 input layers as 28x28 pixels
#1 hidden layer with 100
#10 output units for 0 to 9

import numpy as np
import matplotlib.pyplot as plt
# code enclosed within 2 line of hashes symbol has been used from https://www.python-course.eu/neural_network_mnist.php
# both the training and test data sets have also been obtained from https://www.python-course.eu/neural_network_mnist.php in .csv format
################################################

# code for data processing 

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:/Users/ASUS/Desktop/Virginia TEch/Advanced ML/project/" #replace with your own directory where handwritten digit dataset is stored
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
print(test_data[:10])

test_data[test_data==255]
test_data.shape

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)
    
lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)


train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


################################################

# Sarthak's code

# I chose the first 28000 images from dataset to train the model on
# and 1400 images from dataset to test the model on
n=28000
n_test = 1400

x =    train_imgs[0:n]
x_test = test_imgs[0:n_test]
labels = train_labels_one_hot[0:n]
labels_test = test_labels_one_hot[0:n_test]


def sig(x):

   
    y = 1 / (1 + np.exp(-x))
    y[y==0] = 0.01
    y[y==1] = 0.99
    grad = y * (1 - y)
    grad[grad==0] = 0.01
    grad[grad==1] = 0.99
    return y, grad

no_hidden_layers = 100
w1=np.random.rand(image_pixels, no_hidden_layers)-0.5 #784x100 
w2=np.random.rand( no_hidden_layers, no_of_different_labels )-0.5#100x10 


def CrossEntropy(y_epoch,no,labels):
    prob_y =y_epoch
    
    row_sums = prob_y.sum(axis=1)
    prob_y = prob_y/ row_sums[:, np.newaxis]
    
    return -np.sum(np.multiply(labels,np.log(prob_y)))/(no)

real_values = np.argmax(labels, axis=1)
print('real_values',real_values)
real_values_test = np.argmax(labels_test, axis=1)
#for plotting
plot_CE = []
plot_PE = []
plot_CE_test = []
plot_PE_test = []

no_iter = 3000 #epochs
for i in range(no_iter):
    h,grad1 = sig(x.dot(w1)) 
    y,grad2 = sig(h.dot(w2))
    
    h_test,grad_test1 = sig(x_test.dot(w1)) 
    y_test,grad_test2 = sig(h_test.dot(w2))
    
    
    #backprop update weight 2
    f = labels-y
    s = grad2
    error2 = (np.multiply(f,s)) 
    dw2 = h.T.dot(error2)   
    alpha1 = 1/n #learning rate
    w2 = w2 + alpha1*dw2
  
    #backprop update weight 1
    f=error2.dot(w2.T)
    s=grad1
    error1 = (np.multiply(f,s))
    dw1 = x.T.dot(error1)
    alpha2 = 1/n #learning rate
    w1 = w1 + alpha2*dw1
    i+=1
    
    predictions = np.argmax(y, axis=1)
    predictions_test = np.argmax(y_test, axis=1)

    Correct_predicts = (np.sum(np.equal(predictions,real_values)))
    Correct_predicts_test = (np.sum(np.equal(predictions_test,real_values_test)))
    PE = (n-Correct_predicts)*100/n
    PE_test = (n_test-Correct_predicts_test)*100/n_test
    plot_PE.append(PE)
    plot_PE_test.append(PE_test)
    plot_CE.append(CrossEntropy(y,n,labels))
    plot_CE_test.append(CrossEntropy(y_test,n_test,labels_test))
    
#plots
epoch =[i for i in range(no_iter)]    
plt.figure(0)
plt.xlabel('epochs') 
plt.ylabel('C.E') 
plt.title('Cross Entropy vs Epochs') 
plt.scatter(epoch, plot_CE, label= "train", color= "blue", marker= "*", s=10) 
plt.scatter(epoch, plot_CE_test, label= "test", color= "red", marker= "*", s=10) 
plt.legend() 

plt.figure(1)
plt.xlabel('epochs') 
plt.ylabel('P.E') 
plt.title('Percentage Error vs Epochs') 
plt.plot(epoch, plot_PE, label= "train", color='blue', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=2) 
plt.plot(epoch, plot_PE_test, label= "test", color='red', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='red', markersize=2) 
plt.legend() 
plt.show() 

