# CS4641_Final_Project
Hi! This is my final project for CS 4641 - Food Classification!

Experimented with basic image processing techniques and tested effectiveness of three algos (Decision Trees, SVM, CNN) on a food dataset, along with analysis of the different algorithms and results.

# Directions
Hi! I hope the following information helps:

The dataset can be obtained here:
- https://www.kaggle.com/vermaavi/food11/
- Download it from kaggle, unzip the file, then combine the data from all three folders together!
- I moved the data from evaluation out first, then training, then testing, so some of the training
  images have "(2)" added at the end, and the testing images have "(3)" at the end.
  by adding the "(2)" and "(3)" at the end of the image names.
- Put the folder named "data" in the same folder as main.py
- I can also provide a Google Drive link if necessary

You can install all of the packages used for this project using PIP.
Simply type in "pip install {package-name}" for each of the following packages!
- scikit-learn
- keras
- tensorflow
- pandas
- opencv-python

To re-run the experiments, simply uncomment the desired functions in the main method.
Then, type "python main.py" into your terminal to run the python file.
All of the outputs in my report will be automatically printed by the functions!

1. load_data(x_pixels, y_pixels) - Loads and splits our data into training and testing
2. test_image_preprocessing() - Runs gridsearch on the training data to determine parameters for preprocessing

3. forests_tuning(train_X, train_Y) - Runs k-fold gridsearch with rf hyperparameters on the training data
4. SVM_tuning(train_X, train_Y) - Runs k-fold gridsearch with SVM hyperparameters on the training data
5. neural_networks_tuning() - Runs k-fold gridsearch with neural net hyperparameters on the training data

6. evaluation_random_forests(test_X, test_Y) - Runs scikit's k-fold cross_val_score on the testing data, using the optimal hyperparameters
7. evaluation_svm(test_X, test_Y) - Runs scikit's k-fold cross_val_score on the testing data, using the optimal hyperparameters
8. evaluation_neural_net() - Runs scikit's k-fold cross_val_score on the testing data, using the optimal hyperparameters

