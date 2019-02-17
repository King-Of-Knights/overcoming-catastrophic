from keras.layers.core import Dense
from keras.datasets import mnist
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from util import computer_fisher, ewc_reg

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Original MNIST for Task A
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# Random pixel permutation for Task B
ind = np.arange(x_train.shape[1])
np.random.shuffle(ind)
x_train1 = x_train[:, ind]
x_test1 = x_test[:, ind]
y_train1 = y_train
y_test1 = y_test

# Display three Tasks Dataset images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
plt.title('Task A')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_train1[0].reshape((28, 28)), cmap='gray')
plt.title('Task B')
plt.axis('off')
plt.show()

# Task A training and save the prior weights for the next Task
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, 100, 10, validation_data=(x_test, y_test))
model.save('MNISTA.h5')

# Compute the Fisher Information for each parameter in Task A
print('Processing Fisher Information...')
I = computer_fisher(model, x_train)
print('Processing Finish!')

# Task B EWC training
model_ewcB = Sequential()
model_ewcB.add(Dense(128, activation='relu', input_dim=784, kernel_regularizer=ewc_reg(I[0], model.weights[0]),
                 bias_regularizer=ewc_reg(I[1], model.weights[1])))
model_ewcB.add(Dense(10, activation='softmax', kernel_regularizer=ewc_reg(I[2], model.weights[2]),
                 bias_regularizer=ewc_reg(I[3], model.weights[3])))
model_ewcB.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model_ewcB.load_weights('MNISTA.h5')
model_ewcB.fit(x_train1, y_train1, 100, 10, validation_data=(x_test1, y_test1))

# Task B no penalty training
model_NoP_B = Sequential()
model_NoP_B.add(Dense(128, activation='relu', input_dim=784))
model_NoP_B.add(Dense(10, activation='softmax'))
model_NoP_B.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model_NoP_B.load_weights('MNISTA.h5')
model_NoP_B.fit(x_train1, y_train1, 100, 10, validation_data=(x_test1, y_test1))

# Current Task Performance
B_EWC = 100 * model_ewcB.evaluate(x_test1, y_test1, verbose=0)[1]
B_No_P = 100 * model_NoP_B.evaluate(x_test1, y_test1, verbose=0)[1]
# Previous Task Performance
A_EWC = 100 * model_ewcB.evaluate(x_test, y_test, verbose=0)[1]
A_No_P = 100 * model_NoP_B.evaluate(x_test, y_test, verbose=0)[1]

print("Task A Original Accuracy: %.2f%%" % (100 * model.evaluate(x_test, y_test)[1]))
print("Task B EWC method penalty Accuracy: %.2f%%" % B_EWC)
print("Task B SGD method Accuracy: %.2f%%" % B_No_P)
print("Task A EWC method penalty Accuracy: %.2f%%" % A_EWC)
print("Task A SGD method Accuracy: %.2f%%" % A_No_P)

x = 0
total_width, n = 0.1, 2
width = total_width / n
x = x - (total_width - width) / 2
plt.style.use('ggplot')
plt.bar(x, B_EWC, width=width, label='EWC Task B', hatch='w/', ec='w')
plt.bar(x + width, B_No_P, width=width, label='SGD Task B', hatch='w/', ec='w')
plt.bar(x + 3.5 * width, A_EWC, width=width, label='EWC Task A', hatch='w/', ec='w')
plt.bar(x + 4.5 * width, A_No_P, width=width, label='SGD Task A', hatch='w/', ec='w')
plt.legend(facecolor='white')
plt.xticks(np.array([0., 3.5 * width]), ('Current', 'Previous'))
plt.title('EWC method vs SGD method on \n Current task and Previous task')
plt.xlim(-0.15, 0.35)
plt.ylim(0., 105.)
plt.show()
