import random
import numpy as np
from sklearn import preprocessing
import math

"""
Andrew McCann
Machine Learning 445
Homework #2
Neural Network Assignment
"""

LEARNING_RATE = .3
ACCURACY_CHANGE = .0005
MOMENTUM = .3
WEIGHT_UPPER_BOUND = .25
WEIGHT_LOWER_BOUND = -WEIGHT_UPPER_BOUND
NUM_HIDDEN_UNITS = 100
NUM_FEATURES = 16
EPOCHS = 10


class HiddenPerceptron:
    def __init__(self):
        self.weights = []
        for i in range(NUM_FEATURES):
            self.weights.append(random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND))

        self.bias = random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND)
        self.previous_weight_change = []
        for i in range(NUM_FEATURES):
            self.previous_weight_change.append(0.0)
        self.delta_bias = 0.0

    # def test(self, test_set):

    def forward_prop(self, training_set):

        result = 0
        for i in range(len(training_set)):
            result += self.weights[i] * training_set[i]
        result += self.bias
        """
        print(self.weights)
        print(training_set[1:])
        result = sum(self.weights * training_set[1:]) + self.bias
        # print("Hidden forward_prop result: %f" % result)
        """
        result = sigmoid(result)
        # print("Hidden forward_prop sigmoid result: %.2f" % result)

        return result

        # Pass this output to PerceptronManager to assemble into array

    def back_prop(self, hidden_error, feature, feature_index):
        delta_weight = LEARNING_RATE * hidden_error * feature + MOMENTUM * self.previous_weight_change[feature_index]
        self.weights[feature_index] += delta_weight
        self.previous_weight_change[feature_index] = delta_weight

    def back_prop_bias(self, error):
        delta_bias = LEARNING_RATE * error + MOMENTUM * self.delta_bias
        self.bias += delta_bias
        self.delta_bias = delta_bias


class OutputPerceptron:
    """Individual OutputPerceptron object.

    I made each perceptron an object to allow for more object oriented behavior
    within my code. My unfamiliarity with Python made this sort of a mish-mash
    of brute force methods and numpy ease.

    Attributes:
        letter: A float value representing the "1" output of this perceptron
        weights: array of NUM_FEATURES weights randomized on initialization.
        bias: weight representing the bias
    """

    def __init__(self, letter):
        """Inits Perceptron class with its letter values [0.0,25.0]."""
        self.letter = float(letter)
        self.weights = []
        for i in range(NUM_HIDDEN_UNITS):
            self.weights.append(random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND))
        self.bias = random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND)

        self.error = 0

    def test(self, test_set):
        """Runs an instance of test data against its weight and returns that value.

        Args:
            test_set: One row of NumPy matrix data set that will needs to be tested

        Returns:

        """

    def forward_prop(self, hidden_layer_output, target_letter):
        result = 0
        for i in range(len(hidden_layer_output)):
            result += self.weights[i] * hidden_layer_output[i]
        result += self.bias
        result = sigmoid(result)
        # print("Forward_Prop Target Letter: %f" % target_letter)
        # print(self.letter)
        if target_letter == self.letter:
            return result, .9
        else:
            return result, .1

    def learn(self, params, target_value):
        for i in range(NUM_FEATURES):
            self.weights[i] += (LEARNING_RATE * params[i] * target_value)
        # Apply same method to bias
        self.bias += (LEARNING_RATE * target_value)

    def back_prop(self, output_error, hidden_output):
        for i in range(len(self.weights)):
            self.weights[i] += LEARNING_RATE * output_error * hidden_output[i]

    def back_prop_bias(self, error):
        self.bias += LEARNING_RATE * error  # implied * 1 to rep. bias node.


class PerceptronManager:
    def __init__(self):
        """Instantiates output_perceptron_list and other values."""
        self.output_perceptron_list = []
        self.hidden_perceptron_list = []
        self.hidden_layer_output = []
        self.output_layer_output = []

        # Saving the STD and Mean from training data to use against the test data
        self.scaler = []

        for i in range(26):
            self.output_perceptron_list.append(OutputPerceptron(i))

        for j in range(NUM_HIDDEN_UNITS):
            self.hidden_perceptron_list.append(HiddenPerceptron())

        # Training variables
        self.accuracy_previous_epoch = 0.0
        self.accuracy_current_epoch = 0.0
        self.delta_accuracy = 1

        # Testing counts
        self.overall_accuracy = 0
        self.final_correct = 0
        self.final_iterations = 0

    def randomize_weights(self):
        """Simple little method to re-randomize weights."""
        for i in range(325):
            self.output_perceptron_list[i].randomize()

    # Loop to control duration of epochs
    def epoch_loop(self):
        """Control loop for running training data.

        This function pulls the data from file, converts its
        alphabetical letters to float representations, converts all the parameters
        to float, and then divides by 15 to keep the weights small and easy to manage.
        The loop portion passes the array of data to each perceptron (could probably pass
        specific indices to cut down). For each iteration of the loop the
        perceptron builds it test set and shuffles it each time for higher
        accuracy. This method overall will be more accurate because it waits
        for low-accuracy perceptrons to attain more accurate weights.
        """
        num_epochs = 0

        # Run Perceptron Training Algorithm
        file_data = np.genfromtxt('training.txt', delimiter=',', dtype='O')

        # Convert to numerical value letter instead of Char
        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
            # Convert to floats
        file_data = file_data.astype(np.float32)

        # Save scaling data to apply to test set and transform training data
        self.scaler = preprocessing.StandardScaler().fit(file_data[:, 1:])
        file_data[:, 1:] = self.scaler.transform(file_data[:, 1:])

        # Set total amount of test sets for computing accuracy
        total = len(file_data)

        # Save these once rather than use local calls repeatedly
        l_hidden = len(self.hidden_perceptron_list)
        l_output = len(self.output_perceptron_list)

        # For each training example we must iterate through the entire set of perceptrons
        for e in range(EPOCHS):
            correct = 0
            self.accuracy_current_epoch = 0.0
            for t in range(len(file_data)):

                # Unsure if saving this up here does anything,
                # target_val = 0.0
                training_set = file_data[t]
                target_letter = training_set[0]

                # Reset the output
                self.hidden_layer_output = []
                self.output_layer_output = []

                output_layer_error = []
                hidden_layer_error = []
                # Push the training set through to the hidden layer
                for i in range(l_hidden):
                    # Grab outputs from the hidden layer
                    self.hidden_layer_output.append(self.hidden_perceptron_list[i].forward_prop(training_set[1:]))

                # Push the output from hidden to each output perceptron
                for i in range(l_output):
                    # Get output locally to append later, change target_val
                    output_k, target_val = self.output_perceptron_list[i].forward_prop(self.hidden_layer_output, target_letter)

                    self.output_layer_output.append(output_k)
                    output_layer_error.append((lambda x: x * (1 - x) * (target_val - x))(self.output_layer_output[i]))

                # Get prediction value, NOTE: Argmax returns first instance if tied.
                # Not very concerned with this since the float values are all random
                predicted_letter = np.argmax(self.output_layer_output)

                # Increment accuracy tracker
                if predicted_letter == target_letter:
                    correct += 1

                # HIDDEN LAYER = delta_j = output_j(1- output_j)(sum(k_value_weight * delta_k)
                for j in range(l_hidden):
                    output_layer_sum = []
                    for h in range(l_output):
                        output_layer_sum.append((self.output_perceptron_list[h].weights[j] * output_layer_error[h]))
                    # e = self.hidden_layer_output[j]
                    # error = e * (1-e) * (sum(output_layer_sum))
                    l = lambda x: x * (1 - x) * (sum(output_layer_sum))
                    hidden_layer_error.append(l(self.hidden_layer_output[j]))

                # Back Propagation segment
                # for j in range(l_hidden):
                for i in range(l_output):
                    self.output_perceptron_list[i].back_prop(output_layer_error[i], self.hidden_layer_output)
                    self.output_perceptron_list[i].back_prop_bias(output_layer_error[i])
                # Critical to position this before the regular weight change, fuck
                for i in range(l_hidden):
                    self.hidden_perceptron_list[i].back_prop_bias(hidden_layer_error[i])
                for h in range(NUM_FEATURES):
                    for i in range(l_hidden):
                        self.hidden_perceptron_list[i].back_prop(hidden_layer_error[i], training_set[h + 1], h)

            # Need to insert test data accuracy in here for graphing in report

            # track some accuracy
            print("Epoch: %d, Accuracy: %.2f" % (e +1, 100 * (correct / float(total))))

        return EPOCHS

    def test_accuracy(self):
        file_data = np.genfromtxt('test.txt', delimiter=',', dtype='O')

        # Convert to numerical value letter instead of Char
        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
        # Convert to floats
        file_data = file_data.astype(np.float32)

        # Used to define mean and std
        # self.scaler = preprocessing.StandardScaler().fit_transform(file_data[:, 1:])
        # file_data[:, 1:] = self.scaler

        total = len(file_data)

        l_hidden = len(self.hidden_perceptron_list)
        l_output = len(self.output_perceptron_list)

    def test(self):
        """Method runs test data over trained perceptrons.

        Tally results using testing variables in object to then create
        confusion matrix. No return type needed since its all within class.

        """

        # Generate data set
        file_data = np.genfromtxt('test.txt', delimiter=',', dtype='O')

        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
        file_data = file_data.astype(np.float32)  # Convert to floats
        file_data[:, 1:] = file_data[:, 1:] / 15.0  # Get smaller values for the parameters

        # Create Confusion matrix
        c_matrix = np.zeros(shape=(26, 26), dtype=int)

        # Control data flow and track actual letter
        for i in range(len(file_data)):
            vote_tracking = []
            for j in range(26):
                vote_tracking.append(0.0)
            actual_letter = file_data[i, 0]

            for perceptron in self.output_perceptron_list:
                predicted_letter = perceptron.test(file_data[i, 1:])  # Pass one row at a time, minus actual letter
                vote_tracking[predicted_letter] = ((vote_tracking[predicted_letter]) + 1)

            predicted_letter = np.argmax(vote_tracking)

            # Count number of test instance
            self.final_iterations += 1

            # Confusion Matrix
            c_matrix[int(actual_letter), int(predicted_letter)] += 1

            # Tally for accuracy
            if predicted_letter == actual_letter:
                self.final_correct += 1

        print(c_matrix)

    def menu(self):
        """Lame little menu function I threw together."""
        print(WEIGHT_UPPER_BOUND)
        print(WEIGHT_LOWER_BOUND)
        answer = 1
        while answer != 4:
            print("Ghetto little menu")

            print("Press 1 to train")
            print("Press 2 to test")
            print("Press 4 to quit")
            answer = int(input("Choice: "))

            if answer == 1:
                print("Completed %d epochs" % self.epoch_loop())

            if answer == 2:
                self.test()
                print("Overall accuracy of test is : %d / %d" % (self.final_correct, self.final_iterations))


def sigmoid(result):
    """Sigmoid function."""
    result = 1.0 / (1.0 + float(math.exp(-result)))
    return result


network = PerceptronManager()
network.menu()
