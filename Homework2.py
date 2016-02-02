import random
import numpy as np
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt

"""
Andrew McCann
Machine Learning 445
Homework #2
Neural Network Assignment
"""

LEARNING_RATE = .3
MOMENTUM = .3
WEIGHT_UPPER_BOUND = .25
WEIGHT_LOWER_BOUND = -WEIGHT_UPPER_BOUND
NUM_HIDDEN_UNITS = 4
NUM_FEATURES = 16
EPOCHS = 50


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

    def test(self, test_set):
        result = 0.0
        for i in range(len(test_set)):
            result += self.weights[i] * test_set[i]
        result += self.bias
        result = sigmoid(result)

        return result

    def forward_prop(self, training_set):

        result = 0.0
        for i in range(len(training_set)):
            result += self.weights[i] * training_set[i]
        result += self.bias
        result = sigmoid(result)

        return result

        # Pass this output to PerceptronManager to assemble into array

    def back_prop(self, hidden_error, feature, feature_index):
        delta_weight = LEARNING_RATE * hidden_error * feature + MOMENTUM * self.previous_weight_change[feature_index]
        self.weights[feature_index] += delta_weight
        self.previous_weight_change[feature_index] = delta_weight

    def back_prop_bias(self, error):
        delta_bias = (LEARNING_RATE * error) + (MOMENTUM * self.delta_bias)
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
        self.previous_weight_change = []
        for i in range(NUM_HIDDEN_UNITS):
            self.previous_weight_change.append(0.0)
        self.delta_bias = 0.0

    def test(self, hidden_layer_output):
        """Runs an instance of test data against its weight and returns that value.

        Args:


        Returns:

        """
        result = 0.0
        for i in range(len(hidden_layer_output)):
            result += self.weights[i] * hidden_layer_output[i]
        result += self.bias
        result = sigmoid(result)

        return result



    def forward_prop(self, hidden_layer_output, target_letter):
        result = 0.0
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

    def back_prop(self, output_error, hidden_output):
        for i in range(len(self.weights)):
            delta_weight = LEARNING_RATE * output_error * hidden_output[i] + MOMENTUM * self.previous_weight_change[i]
            self.weights[i] += delta_weight
            self.previous_weight_change[i] = delta_weight

    def back_prop_bias(self, error):
        delta_bias = (LEARNING_RATE * error) + (MOMENTUM * self.delta_bias)
        self.bias += delta_bias
        self.delta_bias = delta_bias

class PerceptronManager:
    def __init__(self):
        """Instantiates output_perceptron_list and other values."""
        self.output_perceptron_list = []
        self.hidden_perceptron_list = []
        #self.hidden_layer_output = []
        #self.output_layer_output = []

        for i in range(26):
            self.output_perceptron_list.append(OutputPerceptron(i))

        for j in range(NUM_HIDDEN_UNITS):
            self.hidden_perceptron_list.append(HiddenPerceptron())

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
        training_accuracy = []
        test_accuracy = []

        # Run Perceptron Training Algorithm
        file_data = np.genfromtxt('training.txt', delimiter=',', dtype='O')
        test_data = np.genfromtxt('test.txt', delimiter=',', dtype='O')

        # Convert to numerical value letter instead of Char
        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
            test_data[i, 0] = ord(test_data[i, 0]) - 65.

            # Convert to floats
        file_data = file_data.astype(np.float32)
        test_data = test_data.astype(np.float32)


        # Save scaling data to apply to test set and transform training data
        self.scaler = preprocessing.StandardScaler().fit(file_data[:, 1:])

        file_data[:, 1:] = self.scaler.transform(file_data[:, 1:])

        test_data[:, 1:] = self.scaler.transform(test_data[:, 1:])


        # Set total amount of test sets for computing accuracy
        total = len(file_data)

        # Save these once rather than use local calls repeatedly
        l_hidden = len(self.hidden_perceptron_list)
        l_output = len(self.output_perceptron_list)

        # For each training example we must iterate through the entire set of perceptrons
        for e in range(EPOCHS):
            correct = 0
            self.accuracy_current_epoch = 0.0
            for t in range(total):

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
            training_accuracy.append(100 * (correct / float(total)))
            test_accuracy.append(self.test_accuracy(test_data))

        return training_accuracy, test_accuracy

    def test_accuracy(self, file_data):


        test_length = len(file_data)
        correct = 0.0


        for t in range(test_length):
            training_set = file_data[t]
            target_letter = training_set[0]

            l_hidden = len(self.hidden_perceptron_list)
            l_output = len(self.output_perceptron_list)

            hidden_layer_output = []
            output_layer_output = []

            for i in range(l_hidden):
                # Get outputs from hidden layer to send through
                hidden_layer_output.append(self.hidden_perceptron_list[i].test(training_set[1:]))
            for i in range(l_output):
                # Collect output to find result.
                output_layer_output.append(self.output_perceptron_list[i].test(hidden_layer_output))

            # Get prediction value, NOTE: argmax returns first instance if tied.
            # Not very concerned with this since the float values are all random
            predicted_letter = np.argmax(output_layer_output)
            #print(hidden_layer_output)
            #print(output_layer_output)
            #print(predicted_letter)
            #print(target_letter)

            # Increment accuracy tracker
            if predicted_letter == target_letter:
                correct += 1.0

        print("Test set accuracy: %.2f" % (100 * (correct / float(test_length))))

        return 100 * (correct / float(test_length))

    def menu(self):
        """Lame little menu function I threw together."""
        epoch_guide = []
        for i in range(EPOCHS+1):
            epoch_guide.append(i)
        print("****PARAMS****")
        print("# Hidden Units: %d" % NUM_HIDDEN_UNITS)
        print("Momentum: %f" % MOMENTUM)
        print("Learning rate: %f" % LEARNING_RATE)
        print("Weight range: +/-%f" % WEIGHT_UPPER_BOUND)

        answer = 1
        while answer != 4:
            print("Ghetto little menu")

            print("Press 1 to train")
            #print("Press 2 to test")
            print("Press 4 to quit")
            answer = int(input("Choice: "))

            if answer == 1:
                training, test = self.epoch_loop()
                training.insert(0,0)
                test.insert(0,0)
                print("Completed %d epochs" % EPOCHS)
                plt.plot(epoch_guide, training, epoch_guide, test, linewidth=1.0)
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.axis([0, EPOCHS, 0, 100])
                plt.show()

            #if answer == 2:
                #self.test()
                #print("Overall accuracy of test is : %d / %d" % (self.final_correct, self.final_iterations))


def sigmoid(result):
    """Sigmoid function."""
    result = 1.0 / (1.0 + float(math.exp(-result)))
    return result


network = PerceptronManager()
network.menu()
