import numpy as np
import os
import cv2
import time
import random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import class_weight as cw
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd # experimental
# verify gpu being used
# tf.config.list_physical_devices('GPU')

# Loads the food data
# Compresses images into (x_pixels by y_pixels)
# By default, loads ALL imagages
def load_data(x_pixels, y_pixels, dataAmount = (6643, 10000)):
    data_folder = './data/food11'

    # Gets all the files
    all_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    # shuffle the files
    all_files = np.array(all_files)
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)
    all_files = all_files[indices]

    # Separate into training and testing
    train_files = all_files[:6643]
    test_files = all_files[6643:]

    # Will be modified
    temp_train = []
    temp_test = []
    train_Y = []
    test_Y = []

    # Randomly sample from files
    train_indices = random.sample(range(len(train_files)), dataAmount[0])
    temp_train = np.take(train_files, train_indices)
    test_indices = random.sample(range(len(test_files)), dataAmount[1])
    temp_test = np.take(test_files, test_indices)
    
    # Shuffle the order of our files
    indices = np.arange(temp_train.shape[0])
    np.random.shuffle(indices)
    temp_train = temp_train[indices]

    indices = np.arange(temp_test.shape[0])
    np.random.shuffle(indices)
    temp_test = temp_test[indices]

    # extract labels
    for file in temp_train:
        train_Y.append(int(file[0:file.find("_")]))

    for file in temp_test:
        test_Y.append(int(file[0:file.find("_")]))

    # Convert the image files into arrays!
    train_X = []
    test_X = []

    start = time.perf_counter()
    count = 0
    for file in temp_train:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #### TESTING STUFF ####
        grayscale_image = cv2.blur(grayscale_image, (5, 5))
        # grayscale_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)

        #### TESTING STUFF ###

        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels), interpolation=cv2.INTER_AREA)
        train_X.append(resized_image)
        count += 1
        if (count % 100 == 0):
            print("train_X " + str(count) + "/" + str(len(temp_train)))
    print("train_X " + str(count) + "/" + str(len(temp_train)) + "\n")

    count = 0
    for file in temp_test:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur
        grayscale_image = cv2.blur(grayscale_image, (5, 5))
        # BLUR
        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels))
        test_X.append(resized_image)
        count += 1
        if (count % 100 == 0):
            print("test_X " + str(count) + "/" + str(len(temp_test)))
    print("test_X " + str(count) + "/" + str(len(temp_test)))

    # for i in range(10):
    #     print(temp_train[i])
    #     print(temp_test[i])
    #     cv2.imshow("test", train_X[i])
    #     cv2.waitKey(0)
    #     cv2.imshow("test", test_X[i])
    #     cv2.waitKey(0)
    

    # look at 3 images to make sure labels correct
    # for i in range(20):
    #     cv2.imshow("test", test_X[i])
    #     print(temp_test[i])
    #     cv2.waitKey(0)
    
    # FINALLY, do some preprocessing
    # - reshape n x n pixels into 1 x (nxn) array for classification
    # - We also normalize to have mean 0    
    train_X = np.array(train_X)
    test_X = np.array(test_X)

    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

    train_X = preprocessing.normalize(train_X)
    test_X = preprocessing.normalize(test_X)

    end = time.perf_counter()
    print("Time to load in photos: " + str(end - start) + "\n")

    return train_X, train_Y, test_X, test_Y

# Loads only the training and validation data, in the amounts desired.
# Output is different as well
def custom_load_data(x_pixels, y_pixels, dataAmount, blur_size, blur_mode):
    data_folder = './data/food11'

    # Gets all the files
    all_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    # shuffle the files
    all_files = np.array(all_files)
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)
    all_files = all_files[indices]

    # Separate into training and testing
    train_files = all_files[:6643]
    test_files = all_files[6643:]

    # now from the TRAINING data, seperate into training and validation
    validation_files = train_files[4000:]
    train_files = train_files[:4000]

    # Will be modified
    temp_train = []
    temp_validation = []
    train_Y = []
    validation_Y = []

    # Randomly sample from files
    indices = random.sample(range(len(train_files)), dataAmount[0])
    temp_train = np.take(train_files, indices)
    indices = random.sample(range(len(validation_files)), dataAmount[1])
    temp_validation = np.take(validation_files, indices)
    
    # Shuffle the order of our files
    indices = np.arange(temp_train.shape[0])
    np.random.shuffle(indices)
    temp_train = temp_train[indices]

    indices = np.arange(temp_validation.shape[0])
    np.random.shuffle(indices)
    temp_validation = temp_validation[indices]


    # extract labels
    for file in temp_train:
        train_Y.append(int(file[0:file.find("_")]))

    for file in temp_validation:
        validation_Y.append(int(file[0:file.find("_")]))


    # Convert the image files into arrays!
    train_X = []
    validation_X = []

    for file in temp_train:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #### TESTING STUFF ####s:
        if blur_size < x_pixels:
            if blur_mode == 'Box':
                grayscale_image = cv2.blur(grayscale_image, (blur_size, blur_size))
            elif blur_mode == 'Gaussian':
                grayscale_image = cv2.GaussianBlur(grayscale_image, (blur_size, blur_size), 0)
        #### TESTING STUFF ###



        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels), interpolation=cv2.INTER_AREA)
        train_X.append(resized_image)

    for file in temp_validation:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #### TESTING STUFF ####
        if blur_size < x_pixels:
            if blur_mode == 'Box':
                grayscale_image = cv2.blur(grayscale_image, (blur_size, blur_size))
            elif blur_mode == 'Gaussian':
                grayscale_image = cv2.GaussianBlur(grayscale_image, (blur_size, blur_size), 0)
        #### TESTING STUFF ####




        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels), interpolation = cv2.INTER_NEAREST)
        validation_X.append(resized_image)

    # FINALLY, do some preprocessing
    # - reshape n x n pixels into 1 x (nxn) array for classification
    # - We also normalize to have mean 0    
    train_X = np.array(train_X)
    validation_X = np.array(validation_X)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    validation_X = np.reshape(validation_X, (validation_X.shape[0], validation_X.shape[1] * validation_X.shape[2]))
    train_X = preprocessing.normalize(train_X)
    validation_X = preprocessing.normalize(validation_X)


    return train_X, train_Y, validation_X, validation_Y

def random_forests_main(train_X, train_Y, test_X, test_Y):  
    start = time.perf_counter()

    clf = RandomForestClassifier(verbose = True, n_jobs = -1)
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    print("Accuracy:", accuracy_score(test_Y, predictions))
    print("Classification Report:", classification_report(test_Y, predictions), "\n")

    end = time.perf_counter()
    print("Random Forests time with n = %i and %i by %i images: %d" % (len(train_X), math.sqrt(train_X.shape[1]), math.sqrt(train_X.shape[1]), end-start))

def support_vector_machines_main(train_X, train_Y, test_X, test_Y):
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)
    
    start = time.perf_counter()

    clf = svm.SVC(decision_function_shape='ovr', kernel = 'linear', cache_size=4000)
    clf.fit(train_X, train_Y)
    predictions = clf.predict(train_X)
    # print("Training Accuracy:", accuracy_score(train_Y, predictions))
    # print("Classification Report:", classification_report(train_Y, predictions), "\n")
    predictions = clf.predict(test_X)
    print("Testing Accuracy:", accuracy_score(test_Y, predictions))
    print("Classification Report:", classification_report(test_Y, predictions), "\n")

    end = time.perf_counter()
    print("SVM time with n = %i and %i by %i images: %d" % (len(train_X), math.sqrt(train_X.shape[1]), math.sqrt(train_X.shape[1]), end-start))

def test_image_preprocessing():
    image_sizes = [4, 8, 16, 32, 128, 256]
    data_counts = [100, 1000, 4000]
    blur_sizes = [5, 25]
    blur_modes = ['None', 'Box', 'Gaussian']

    for pixels in image_sizes:
        for n in data_counts:
            for blur_size in blur_sizes:
                for blur_mode in blur_modes:
                    start = time.perf_counter()
                    train_X, train_Y, validation_X, validation_Y = custom_load_data(pixels, pixels, (n, 2643), blur_size, blur_mode)
                    end = time.perf_counter()
                    load_time = end-start

                    clf = RandomForestClassifier(n_jobs = -1)
                    clf.fit(train_X, train_Y)
                    predictions = clf.predict(validation_X)
                    print("Accuracy: %.4f, px: %i, n: %i, bl: %s, blsz: %i" % (accuracy_score(validation_Y, predictions), pixels, n, blur_mode, blur_size))
                    
                    end = time.perf_counter()
                    algo_time = end-start
                    print("Load time: %i, Algo time: %i" % (load_time, algo_time))


# DON'T USE BLUR!
# Also, load 2D images!
# Convert Y to one hot as well
def load_data_neural_nets(x_pixels, y_pixels, dataAmount = (6643, 10000)):
    data_folder = './data/food11'

    # Gets all the files
    all_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    # shuffle the files
    all_files = np.array(all_files)
    indices = np.arange(len(all_files))
    np.random.shuffle(indices)
    all_files = all_files[indices]


    # Separate into training and testing
    train_files = all_files[:6643]
    test_files = all_files[6643:]

    # Will be modified
    temp_train = []
    temp_test = []
    train_Y = []
    test_Y = []

    # Randomly sample from files
    train_indices = random.sample(range(len(train_files)), dataAmount[0])
    temp_train = np.take(train_files, train_indices)
    test_indices = random.sample(range(len(test_files)), dataAmount[1])
    temp_test = np.take(test_files, test_indices)
    
    # Shuffle the order of our files
    indices = np.arange(temp_train.shape[0])
    np.random.shuffle(indices)
    temp_train = temp_train[indices]

    indices = np.arange(temp_test.shape[0])
    np.random.shuffle(indices)
    temp_test = temp_test[indices]

    # extract labels
    for file in temp_train:
        train_Y.append(int(file[0:file.find("_")]))

    for file in temp_test:
        test_Y.append(int(file[0:file.find("_")]))

    # Convert the image files into arrays!
    train_X = []
    test_X = []

    start = time.perf_counter()
    count = 0
    for file in temp_train:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels), interpolation=cv2.INTER_AREA)
        train_X.append(resized_image)
        count += 1
        if (count % 100 == 0):
            print("train_X " + str(count) + "/" + str(len(temp_train)))
    print("train_X " + str(count) + "/" + str(len(temp_train)) + "\n")

    count = 0
    for file in temp_test:
        ## read image
        image = cv2.imread(data_folder + "/" + file)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #resize
        resized_image = cv2.resize(grayscale_image, (x_pixels, y_pixels))
        test_X.append(resized_image)
        count += 1
        if (count % 100 == 0):
            print("test_X " + str(count) + "/" + str(len(temp_test)))
    print("test_X " + str(count) + "/" + str(len(temp_test)))

    # Normalize to have mean 0    
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)

    # reshape
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
    # normalize
    train_X = preprocessing.normalize(train_X)  
    test_X = preprocessing.normalize(test_X)
    # shape back
    train_X = np.reshape(train_X, (train_X.shape[0], x_pixels, y_pixels, 1))
    test_X = np.reshape(test_X, (test_X.shape[0], x_pixels, y_pixels, 1))
    # print(train_X.shape)
    # print(test_X.shape)

    # # Convert Y to one hot!
    # temp_train_Y = np.array(train_Y)
    # # creates np array of 11 0's
    # train_Y = np.zeros((temp_train_Y.size, temp_train_Y.max() + 1))
    # # For every row vector, sets appropriate lement to 1
    # train_Y[np.arange(temp_train_Y.size), temp_train_Y] = 1

    # # Convert Y to one hot!
    # temp_test_Y = np.array(test_Y)
    # test_Y = np.zeros((temp_test_Y.size, temp_test_Y.max() + 1))
    # test_Y[np.arange(temp_test_Y.size), temp_test_Y] = 1
    # # print(sum(test_Y))
    train_Y = to_categorical(train_Y)
    test_Y = to_categorical(test_Y)

    # print(train_Y.shape)
    # print(test_Y.shape)


    end = time.perf_counter()
    print("Time to load in photos: " + str(end - start) + "\n")
    return train_X, train_Y, test_X, test_Y

# uses separate load data function
def neural_networks_main():
    x_pixels = 128
    y_pixels = 128
    dataAmount = (3000,10)

    start = time.perf_counter()

    # dataAmount = (6643, 10000) by default
    train_X, train_Y, test_X, test_Y = load_data_neural_nets(x_pixels, y_pixels, dataAmount)
    print(train_X.shape)
    print(test_X.shape)

    model = create_model(learn_rate = 0.1, momentum = 0.3)

    # # create model    
    # model = Sequential()
    # model.add(Conv2D(64, (3,3), input_shape = train_X.shape[1:]))
    # # CNN and max pooling good
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(64, (3,3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Flatten())
    # model.add(Dense(64))

    # model.add(Dense(11))
    # model.add(Activation('softmax'))
    
    # We need to ask for CATEGORICAL ACCURACY, otherwise will produce absurdly high results when loss = binary_crossentropy
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_accuracy'])
    model.fit(train_X, train_Y, batch_size = 64, epochs=30, validation_split = 0.1)
    
    # Evaluation
    score = model.evaluate(test_X, test_Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    end = time.perf_counter()
    print(end-start)


def SVM_tuning(train_X, train_Y):
    start = time.perf_counter()

    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [.001, .01, .1, 1],'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(svm.SVC(cache_size=2000),param_grid, verbose=2, n_jobs = -1)

    # reduce data down to n = 4000
    train_X = train_X[:4000]
    train_Y = train_Y[:4000]
    grid.fit(train_X, train_Y)
    
    # GOD SEND
    print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    # Best estimator
    print(grid.best_estimator_)
    end = time.perf_counter()
    print("Time: %i" % (end - start))
    return

def forests_tuning(train_X, train_Y):
    num_features = int(train_X.shape[1])
    start = time.perf_counter()
    param_grid = {'n_estimators': [50, 100, 300, 500, 800, 1200], 'max_features': [10, 20, 40, 64, 80]}
    grid = GridSearchCV(RandomForestClassifier(verbose = True, n_jobs = 8),param_grid, verbose=2, n_jobs = 8)
    grid.fit(train_X, train_Y)

    # GOD SEND
    print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    # Best estimator
    print(grid.best_estimator_)
    end = time.perf_counter()
    print("Time: %i" % (end - start))
    return

def create_model(learn_rate = 0.3, batch_size = 24):
    # test
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = (128, 128, 1)))
    # CNN and max pooling good
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(11))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['categorical_accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc'])
    return model


def neural_networks_tuning():
    train_X, train_Y, test_X, test_Y = load_data_neural_nets(128, 128)
    print(train_X.shape)
    print(train_Y.shape)
    
    # Results
    start = time.perf_counter()
    
    model = KerasClassifier(build_fn=create_model, epochs=30, verbose=1)
    # define the grid search parameters
    learn_rate = [0.1, 0.2, 0.3]
    batch_size = [16, 24]
    param_grid = dict(learn_rate=learn_rate, batch_size = batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=3)
    grid.fit(train_X, train_Y)

    # Results
    print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    # Best estimator
    print(grid.best_estimator_)
    end = time.perf_counter()
    print("Time: %i" % (end - start))
    return

def evaluation_random_forests(test_X, test_Y):
    start = time.perf_counter()

    clf = RandomForestClassifier(verbose = True, n_jobs = -1, n_estimators=1200, max_features=20)
    scores = cross_val_score(clf, test_X, test_Y, cv = 5, n_jobs = -1)
    print(scores)

    #t_value for df = 4 and 95% confidence interval
    t_value = 2.131802
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() / math.sqrt(5) * t_value))

    end = time.perf_counter()
    print("Evaluation time: %i" % (end-start))

def evaluation_svm(test_X, test_Y):
    # svm just takes too much time
    test_X = test_X[:4000]
    test_Y = test_Y[:4000]
    start = time.perf_counter()
    clf = svm.SVC(C = 10, kernel = 'rbf', gamma = 1, decision_function_shape='ovr', cache_size=4000)
    scores = cross_val_score(clf, test_X, test_Y, cv = 5, n_jobs=-1)

    print(scores)

    #t_value for df = 4 and 95% confidence interval
    t_value = 2.131802
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() / math.sqrt(5) * t_value))
    end = time.perf_counter()
    print("Evaluation time: %i" % (end-start))

def evaluation_neural_net():
    train_X, train_Y, test_X, test_Y = load_data_neural_nets(128, 128,dataAmount =(10,10000))

    # Results
    start = time.perf_counter()
    
    model = KerasClassifier(build_fn=create_model, epochs=30, verbose=1)
    scores = cross_val_score(model, test_X, test_Y, cv = 5)

    print(scores)

    #t_value for df = 4 and 95% confidence interval
    t_value = 2.131802
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() / math.sqrt(5) * t_value))
    end = time.perf_counter()
    print("Evaluation time: %i" % (end-start))

if __name__ == '__main__':
    # Seed to replicate results (lucky number 7)
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)

    # Set desired image size and # of points (train, test)
    x_pixels = 64
    y_pixels = 64
    # dataAmount = (5, 1000) # (# train points, # test points)
    
    ## TESTING IMAGE PREPROCESSING
    # test_image_preprocessing()

    # Loading data
    train_X, train_Y, test_X, test_Y = load_data(x_pixels, y_pixels)

    ######## TUNING HYPERPARAMETERS #######
    # forests_tuning(train_X, train_Y)
    # SVM_tuning(train_X, train_Y)
    # neural_networks_tuning()


    ######## USING TUNED HYPER PARAMETERS ON TESTING DATA #######
    # evaluation_random_forests(test_X, test_Y)
    # evaluation_svm(test_X, test_Y)
    # evaluation_neural_net()