#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np 
from tqdm import tqdm
import cv2
import os
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
init_notebook_mode(connected=True)
RANDOM_SEED = 123


# In[2]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[3]:


TRAIN_DIR = train_datagen.flow_from_directory(r"C:\Users\KIIT\Downloads\archive (4)\Training\Training",
                                                 target_size=(48, 48),
                                                 batch_size=15,
                                                 class_mode='categorical')


# In[ ]:





# In[4]:



TEST_DIR = test_datagen.flow_from_directory(r"C:\Users\KIIT\Downloads\archive (4)\Testing\Testing",
                                            target_size=(48, 48),
                                            batch_size=10,
                                            class_mode='categorical',
                                            shuffle=False)

IMG_SIZE = (48, 48)


# In[5]:


def load_data(generator, IMG_SIZE):
    X = []
    y = []
    i = 0
    labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    
    dir_path = generator.directory  # Get the directory path from the generator object
    
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            label_dir = os.path.join(dir_path, path)  # Get the complete path for the emotion label directory
            for file in os.listdir(label_dir):
                if not file.startswith('.'):
                    img = cv2.imread(os.path.join(label_dir, file)) 
                    img = img.astype('float32') / 255
                    resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
                    X.append(resized)
                    y.append(i)
            i += 1
    
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)  # Pass the generator object instead of TRAIN_DIR
print(train_labels)


# In[ ]:





# In[6]:



X_test, y_test, test_labels = load_data(TEST_DIR, IMG_SIZE)


# In[ ]:





# In[ ]:





# In[7]:


def plot_samples(X, y, labels_dict, n=50):
   
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(10,3))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle(labels_dict[index])
        plt.show()


# In[8]:


plot_samples(X_train, y_train, train_labels, 10)


# In[9]:


from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=6)
Y_train.shape


# In[10]:


Y_test = to_categorical(y_test, num_classes=6)
Y_test.shape


# In[11]:


from keras.applications.vgg19 import VGG19

base_model = VGG19(
        weights=None,
        include_top=False, 
        input_shape=IMG_SIZE + (3,)
    )

base_model.summary()


# In[12]:


NUM_CLASSES = 6

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(NUM_CLASSES, activation="softmax"))


# In[13]:


def deep_model(model, X_train, Y_train, epochs, batch_size):
   
    model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy'])
    
#     es = EarlyStopping(monitor='val_loss',
#                            restore_best_weights=True,
#                            mode='min'
#                           min_delta=1.5)
    history = model.fit(X_train
                       , Y_train
                       , epochs=epochs
                       , batch_size=batch_size
                       , verbose=1)
    return history


# In[14]:


epochs = 40
batch_size = 128

history = deep_model(model, X_train, Y_train, epochs, batch_size)


# In[15]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm,2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[16]:


predictions = model.predict(X_test)
y_pred = [np.argmax(probas) for probas in predictions]


accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_test, y_pred) 
cm = plot_confusion_matrix(confusion_mtx, classes = list(test_labels.items()), normalize=False)


# In[17]:


new_predictions = model.predict(X_test)
y_pred = [np.argmax(probas) for probas in new_predictions]
y_pred = [test_labels[k] for k in y_pred]


# In[18]:


filenames = TEST_DIR.filenames
actual_class = [test_labels[h] for h in TEST_DIR.classes]


# In[19]:


import pandas as pd

pred_result = pd.DataFrame({"Filename":filenames,
                           "Predictions":y_pred,
                           "Actual Values":actual_class})

pred_result.head()


# In[20]:


from random import randint


# In[21]:


from random import randint
import os
import cv2
import matplotlib.pyplot as plt

base_path = TEST_DIR.directory  # Get the directory path from TEST_DIR
num_rows = len(pred_result)

for i in range(30):  # Loop for 30 images
    rnd_number = randint(0, num_rows - 1)
    filename, pred_class, actual_class = pred_result.loc[rnd_number]

    img_path = os.path.join(base_path, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Predicted Class: {} {} Actual Class: {}".format(pred_class, '\n', actual_class))
    plt.show()


# In[ ]:





# In[ ]:


from sklearn.metrics import classification_report

# Evaluation on the test dataset
y_pred = model.predict(TEST_DIR)
y_pred = np.argmax(y_pred, axis=1)
y_true = TEST_DIR.classes

# Classification report
print(classification_report(y_true, y_pred))


# In[ ]:


import joblib

# Assuming you have already trained and obtained your model
# For this example, let's assume your model object is named 'model'

# Save the model to a file
file_path = r'C:\Users\KIIT\OneDrive\Desktop\dataset3\model.h5'

joblib.dump(model, file_path)

print(f"Model saved to {file_path}")


# In[ ]:





# In[ ]:





# In[ ]:




