import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pandas as pd
import matplotlib.pyplot as plt
import random

TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
VALIDATE_DIR = 'dataset/validate'
IMG_SIZE = 96   #real size 224 x 224
LR = 1e-3

MODEL_NAME = 'skinbenignvsmalign-{}-5{}.model'.format(LR, 'CNN')

def label_img(img):
  word_label = img.split('.')
  if str(word_label[0]) == 'benign':
    return [1,0]
  elif str(word_label[0]) == 'malign':
    return [0,1]

def brightness_augment(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def data_augmentation(img):
  augmented_images = []

  augmented_images.append(np.rot90(img))  #rotate 90 degrees
  augmented_images.append(np.rot90(img, 2)) #rotate 180 degrees
  augmented_images.append(np.rot90(img, 3)) #rotate 270 degrees
  augmented_images.append(np.fliplr(img)) #flip image
  augmented_images.append(brightness_augment(img, 0.7))  #darker image
  augmented_images.append(brightness_augment(img, 1.15))  #lighter image
  augmented_images.append(sp_noise(img, 0.005))	#add salt pepper noise

  return augmented_images


def create_train_data():
  training_data = []
  augmented_data = []
  for img in tqdm(os.listdir(TRAIN_DIR)):
    label = label_img(img)
    path = os.path.join(TRAIN_DIR, img)
    src = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    training_data.append([np.array(img), np.array(label)])
    augmented_data = data_augmentation(img)
    for augmented_image in augmented_data:
      training_data.append([np.array(augmented_image), np.array(label)])
  shuffle(training_data)
  np.save('train_data.npy', training_data)
  return training_data

def create_test_data():
  testing_data = []
  augmented_data = []
  for img in tqdm(os.listdir(TEST_DIR)):
    label = label_img(img)
    path = os.path.join(TEST_DIR, img)
    src = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    testing_data.append([np.array(img), np.array(label)])
    augmented_data = data_augmentation(img)
    for augmented_image in augmented_data:
      testing_data.append([np.array(img), np.array(label)])
  shuffle(testing_data)
  np.save('test_data.npy', testing_data)
  return testing_data

def create_validate_data():
  validating_data = []
  for img in tqdm(os.listdir(VALIDATE_DIR)):
    label = label_img(img)
    path = os.path.join(VALIDATE_DIR, img)
    src = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    validating_data.append([np.array(img), np.array(label)])
  shuffle(validating_data)
  np.save('validation_data.npy', validating_data)
  return validating_data

def create_convolutional_neural_net():
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model

def train_convolutional_neural_net(model, train_data, test_data):
    X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    Y = [i[1] for i in train_data]

    test_x = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_y = [i[1] for i in test_data]

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, batch_size=32, shuffle=True, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

    return model

def show_validation(validate_data, model):
  fig = plt.figure()
  for num, data in enumerate(validate_data[:16]):
    #Benign: [1,0]
    #Malign: [0,1]
    img_label = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num+1)

    model_out = model.predict([img_data])[0]

    if(img_label[0] == 1):
      real_label = 'REAL: Benign'
    else:
      real_label = 'REAL: Malign'

    if(model_out[1] > model_out[0]):
      predicted_label = ' PREDICTED: Malign'
    else:
      predicted_label = ' PREDICTED: Benign'

    label = predicted_label + ' |||| ' + real_label + '\n' + str(model_out)

    y.imshow(img_data)
    y.set_title(label, fontsize=8)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
  plt.show()

def show_confusion_matrix_validation(validate_data, model):
  malign_malign = 0
  benign_malign = 0
  malign_benign = 0
  benign_benign = 0
  for data in validate_data:
    img_data = data[0]
    img_label = data[1]

    model_out = model.predict([img_data])[0]

    if(img_label[0] == 1):
      is_malign = False
    else:
      is_malign = True

    if(model_out[1] > model_out[0]):
      predicted_malign = True
    else:
      predicted_malign = False

    if(is_malign and predicted_malign):
      malign_malign += 1
    elif(not is_malign and predicted_malign):
      benign_malign += 1
    elif(is_malign and not predicted_malign):
      malign_benign += 1
    else:
      benign_benign += 1

  confusion_matrix = {'': ['Malign', 'Benign'], 'Malign': [malign_malign, benign_malign], 'Benign': [malign_benign, benign_benign]}
  dataframe_confusion_matrix = pd.DataFrame(data=confusion_matrix)

  print()
  print('CONFUSION MATRIX')
  print()
  print(dataframe_confusion_matrix)
  print()

  total_accuracy = (benign_benign + malign_malign) / (benign_malign + malign_benign + malign_malign + benign_benign)
  malign_accuracy = malign_malign / (malign_malign + malign_benign)
  benign_accuracy = benign_benign / (benign_benign + benign_malign)

  print('TOTAL ACCURACY:', total_accuracy, sep= ' ')
  print('ACCURACY MALIGN:', malign_accuracy)
  print('ACCURACY BENIGN:', benign_accuracy)


def main():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    if(os.path.exists('train_data.npy') and os.path.exists('test_data.npy')):
        train_data = np.load('train_data.npy')
        print('Training data loaded...')
        test_data = np.load('test_data.npy')
        print('Testing data loaded...')
    else:
        train_data = create_train_data()
        print('Training data created...')
        test_data = create_test_data()
        print('Testing data created...')

    model = create_convolutional_neural_net()

    if(os.path.exists('{}.meta'.format(MODEL_NAME))):
        model.load(MODEL_NAME)
        print('Model loaded...')
    else:
        model = train_convolutional_neural_net(model, train_data, test_data)
        #tensorboard --logdir=foo:C:\Users\unai1\Desktop\IA\Cancer_Project\log

    if(os.path.exists('validation_data.npy')):
        validate_data = np.load('validation_data.npy')
        print('Validation data loaded...')
    else:
        validate_data = create_validate_data()
        print('Validation data created...')

    show_validation(validate_data, model)
    show_confusion_matrix_validation(validate_data, model)
    

if __name__ == "__main__":
    main()
