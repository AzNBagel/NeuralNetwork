import random
import numpy as np
from sklearn import preprocessing


"""
Andrew McCann
Machine Learning 445
Homework #2
Neural Network Assignment
"""

LEARNING_RATE = .2
ACCURACY_CHANGE = .0005
MOMENTUM = .03
WEIGHT_UPPER_BOUND = .25
WEIGHT_LOWER_BOUND = -WEIGHT_UPPER_BOUND
NUM_HIDDEN_UNITS = 4
NUM_FEATURES = 16


class HiddenPerceptron:

    def __init__(self):
        self.weights = []
        for i in range(NUM_FEATURES):
            self.weights.append(random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND))

        self.bias = random.uniform(WEIGHT_LOWER_BOUND, WEIGHT_UPPER_BOUND)
        self.previous_weight_change = 0

    # def test(self, test_set):

    def forward_prop(self, training_set):
        result = sum(self.weights * training_set[1:]) + self.bias
        result = sigmoid(result)
        return result

        # Pass this output to PerceptronManager to assemble into array
    def back_prop(self, hidden_error, feature, feature_index):
        self.weights[feature_index] += MOMENTUM * hidden_error * feature_index


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
        self.letter = letter
        self.weights = []
        for i in range(NUM_HIDDEN_UNITS):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

    def test(self, test_set):
        """Runs an instance of test data against its weight and returns that value.

        Args:
            test_set: One row of NumPy matrix data set that will needs to be tested

        Returns:

        """

    def forward_prop(self, hidden_layer_output):
        result = sum(np.dot(self.weights, hidden_layer_output)) + self.bias
        result = sigmoid(result)
        return result

        # # # # # # DEPRECATED

        """Runs training data against weights and calls learning if necessary.

        The main function of this project. The data_array will have all the
        testing sets for a particular letter. The test_set conversion stacks
        the matrices and then I shuffle them immediately after to simulate
        interleaving the examples for higher accuracy.

        Args:
            data_array: An array of 26 matrices representing data from A-Z

        Returns:
            A tuple of how many iterations on this training data, and how
            many were correct.
        """

        correct = 0.0
        num_trains = 0.0
        test_set = np.vstack((data_array[self.letter], data_array[self.letter_2]))
        np.random.shuffle(test_set)

        for i in range(len(test_set)):
            target_value = test_set[i, 0]

            # Assign the target value of the pair.
            if target_value == self.letter:
                target_value = 1
            else:
                target_value = -1

            # Result is useless by itself here so dump into SGN
            result = sgn(np.dot(self.weights, test_set[i, 1:]) + self.bias)

            # Figure out accuracy
            if result == target_value:
                correct += 1
            else:
                self.learn(test_set[i, 1:], target_value)

            # Track number of iterations
            num_trains += 1
        accuracy = correct/num_trains
        # Pass through the number of iterations plus the num correct
        return accuracy

    def learn(self, params, target_value):
        for i in range(NUM_FEATURES):
            self.weights[i] += (LEARNING_RATE * params[i] * target_value)
        # Apply same method to bias
        self.bias += (LEARNING_RATE * target_value)

    def randomize(self):
        for i in range(NUM_FEATURES):
            self.weights[i] = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)

    def back_prop(self, error, hidden_output, hidden_index):
        self.weights[hidden_index] += MOMENTUM * error * hidden_output

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

        for i in range(NUM_HIDDEN_UNITS):
            self.hidden_layer_output.append(HiddenPerceptron())

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
        file_data = file_data.astype(np.float32)     # Convert to floats

        self.scaler = preprocessing.StandardScaler().fit_transform(file_data[:, 1:])
        file_data[: ,1:] = self.scaler

        """ DEPRECATED
        files_out = []
        for i in range(26):
            temp = file_data[file_data[:, 0] == float(i)]
            files_out.append(temp)
        """

        previous_error = 0
        # For each training example we must iterate through the entire set of perceptrons
        for training_set in range(len(file_data)):
            # Reset the output
            self.hidden_layer_output = []
            self.output_layer_output = []
            previous_hidden_delta = 0
            output_layer_error = []
            hidden_layer_error = []
            # Push the training set through to the hidden layer
            for perceptron in self.hidden_perceptron_list:
                # Grab outputs from the hidden layer
                self.hidden_layer_output.append(perceptron.forward_prop(training_set))
            # Push the output from hidden to each output perceptron
            for perceptron in self.output_perceptron_list:
                # target_letter = training_set[0]

                # Store each output from the output layer to calculate
                target_val, output_k = perceptron.forward_prop(self.hidden_layer_output)
                self.output_layer_output.append(output_k)
            # Calculate error at output first, because used in hidden layer error
            # OUTPUT LAYER = delta_k = output_k(1 - output_k)(target_val - output_k)
            for i in self.output_perceptron_list:
                # made a lambda function because I felt like it
                output_layer_error.append((lambda x: x(1-x)(target_val - x)) (self.output_layer_output[i]))

            # HIDDEN LAYER = delta_j = output_j(1- output_j)(sum(k_value_weight * delta_k)
            for j in self.hidden_perceptron_list:
                output_layer_sum = []
                for h in self.output_perceptron_list:
                    output_layer_sum.append((h.self.weights[j] * output_layer_error[h]))
                hidden_layer_error.append(lambda x: x(1-x)(sum(output_layer_sum))(self.hidden_layer_output[j]))

            # Back prop dat net
            for j in range(len(self.hidden_layer_output)):
                for i in range(len(self.output_perceptron_list)):
                    self.output_perceptron_list[i].back_prop(output_layer_error[i], self.hidden_layer_output[j], j)
            for h in range(1, NUM_FEATURES):
                for i in range(len(hidden_layer_error)):
                    self.hidden_perceptron_list[i].back_prop(hidden_layer_error[i], training_set[h], h)

        return num_epochs

    def test(self):
        """Method runs test data over trained perceptrons.

        Tally results using testing variables in object to then create
        confusion matrix. No return type needed since its all within class.

        """
        # Generate data set
        file_data = np.genfromtxt('test.txt', delimiter=',', dtype='O')

        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
        file_data = file_data.astype(np.float32)     # Convert to floats
        file_data[:, 1:] = file_data[:, 1:] / 15.0     # Get smaller values for the parameters

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
        answer = 1
        while answer != 4:
            print("Ghetto little menu")
            print("Press 1 randomize weights")
            print("Press 2 to train")
            print("Press 3 to test")
            print("Press 4 to quit")
            answer = int(input("Choice: "))

            if answer == 1:
                self.randomize_weights()

            if answer == 2:
                # print("Completed %d epochs" % self.epoch_loop())

            if answer == 3:
                self.test()
                print("Overall accuracy of test is : %d / %d" % (self.final_correct, self.final_iterations))

def sigmoid(result):
    """Sigmoid function."""
    return 1.0/(1.0 +np.exp(-result))

#def sigmoid_prime(result):
#    """Take derivative of the sigmoid function."""
#    # Pretty amazing math trick here from notes.
#    return sigmoid(result)*(1-sigmoid(result))

network = PerceptronManager()
network.menu()
