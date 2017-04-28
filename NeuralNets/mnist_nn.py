# Step 1 : Import necessary Libraries
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
#from keras import backend as K
#K.set_image_dim_ordering('th')
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
 


#%%
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print ("size of train set:",len(train_set))

(X_train,y_train) = (train_set[0],train_set[1])
(X_val,y_val) = (valid_set[0],valid_set[1])
(X_test,y_test) = (test_set[0],test_set[1])

#print "Num of train data:",y_train.shape[0]
#print "Num of validation data:",y_val.shape[0]
#print "Num of test data:",y_test.shape[0]
#
#print "shape of train data:",X_train.shape
#print "shape of validation data:",X_val.shape
#print "shape of test data:",X_test.shape

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0].reshape(28,28), cmap='gray')
plt.subplot(222)
plt.imshow(X_train[1].reshape(28,28), cmap='gray')
plt.subplot(223)
plt.imshow(X_train[2].reshape(28,28), cmap='gray')
plt.subplot(224)
plt.imshow(X_train[3].reshape(28,28), cmap='gray')
# show the plot
plt.show()

#print y_train[0:4]


#num_classes = 10

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

inputShape=X_train[0:1,:].shape
hidden_dim = 784


#step 3: create model
model = Sequential()
model.add(Dense(hidden_dim, input_dim = 784, init='normal', activation='relu'))
#model.add(Dense(hidden_dim, input_dim = 784, kernel_initializer='normal', activation='relu'))

model.add(Dense(num_classes, init='normal', activation='softmax'))
#model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# Step 4: Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=10, batch_size=200, verbose=2)

#hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
(loss,accuracy) = model.evaluate(X_test, y_test, verbose=0)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("Baseline Error: %.2f%%" % (100-accuracy*100))


plt.subplot(221)
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.subplot(222)
plt.imshow(X_test[1].reshape(28,28), cmap='gray')
plt.subplot(223)
plt.imshow(X_test[2].reshape(28,28), cmap='gray')
plt.subplot(224)
plt.imshow(X_test[3].reshape(28,28), cmap='gray')
# show the plot
plt.show()
model.predict_classes(X_test[0:4])

epochs = 10

print (hist.history)

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']

xc=range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


#%%
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)

#p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(0)', 'class 1(1)', 'class 2(2)','class 3(3)',
					'class 4(4)','class 5(5)','class 6(6)','class 7(7)','class 8(8)','class 9(9)']
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix without mean-normalized FER data')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.show()

#%%
