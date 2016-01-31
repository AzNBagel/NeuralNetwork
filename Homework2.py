import random
import numpy as np


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



class HiddenPerceptron:

    def __init__(self):
        self.weights = []
        for i in range(16):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)




    def test(self, test_set):

    def learn(self):


class OutputPerceptron:
    """Individual OutputPerceptron object.

    I made each perceptron an object to allow for more object oriented behavior
    within my code. My unfamiliarity with Python made this sort of a mish-mash
    of brute force methods and numpy ease.

    Attributes:
        letter: A float value representing the "1" output of this perceptron
        weights: array of 16 weights randomized on initialization.
        bias: weight representing the bias
    """

    def __init__(self, letter):
        """Inits Perceptron class with its letter values [0.0,25.0]."""
        self.letter = letter
        self.weights = []
        for i in range(16):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

    def test(self, test_set):
        """Runs an instance of test data against its weight and returns that value.

        Args:
            test_set: One row of NumPy matrix data set that will needs to be tested

        Returns:
            letter_1 will be returned if the sgn() function returns 1, letter_2
            will be returned if the sgn() function returns -1.
        """

        result = sgn(np.dot(self.weights, test_set) + self.bias)
        if result == 1:
            return self.letter
        else:
            return self.letter_2

    def train(self, data_array):
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
        for i in range(16):
            self.weights[i] += (LEARNING_RATE * params[i] * target_value)
        # Apply same method to bias
        self.bias += (LEARNING_RATE * target_value)

    def randomize(self):
        for i in range(16):
            self.weights[i] = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)


class PerceptronManager:
    """Management class to pass and control data flow to perceptrons.

    Mostly a controller class for simplified conversion in later assignments.
    This class controls the start of training routines, can randomize weights,
    or trigger testing. Also includes a terribly implemented menu.

    Attributes:
        output_perceptron_list: array of perceptrons totalling 325 ((n(n-1))/2).
            This is the main object that iterates through for testing and training
        accuracy_previous_epoch: Training, tracks previous accuracy.
        accuracy_current_epoch: Training, tracks current epoch.
        delta_accuracy: Stopping condition, tracks different between current and prev.
        overall_accuracy: Testing, saves overall accuracy of testing set.
        final_correct: Testing, tally of correct predictions.
        final_iterations: Testing, tally of total iterations.
    """

    def __init__(self):
        """Instantiates output_perceptron_list and other values."""
        self.output_perceptron_list = []
        self.hidden_perceptron_list = []
        self.hidden_layer_output = []
        self.output_layer_output = []

        for i in range(26):
            self.output_perceptron_list.append(OutputPerceptron(i))

        for i in range(NUM_HIDDEN_UNITS):
            self.hidden_layer_output.append(HiddenPerceptron())



        # OLD INIT BELOW

        # Loop creates distinct pairings of letter representations.
        for i in range(25):
            for j in range(i+1, 26):
                if i != j:
                    self.output_perceptron_list.append(OutputPerceptron(i, j))

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

        for i in range(len(file_data)):
            file_data[i, 0] = ord(file_data[i, 0]) - 65.
        file_data = file_data.astype(np.float32)     # Convert to floats
        file_data[:, 1:] = file_data[:, 1:] / 15.0     # Get smaller values for the parameters

        # Sort data into
        files_out = []
        for i in range(26):
            temp = file_data[file_data[:, 0] == float(i)]
            files_out.append(temp)

        # Corrected loop to perform epochs
        # Previous iteration defined an epoch as a single round of
        # Testing for all 325 perceptrons.
        # New epoch is defined at the perceptron level, not system.
        for perceptron in self.output_perceptron_list:
            perceptron_delta = 1
            current_accuracy = 0
            while perceptron_delta > ACCURACY_CHANGE:
                previous_accuracy = current_accuracy
                current_accuracy = perceptron.train(files_out)

                perceptron_delta = abs(previous_accuracy - current_accuracy)
                num_epochs += 1

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
                print("Completed %d epochs" % self.epoch_loop())

            if answer == 3:
                self.test()
                print("Overall accuracy of test is : %d / %d" % (self.final_correct, self.final_iterations))

def sgn(result):
    if result < 0:
        return -1
    else:
        return 1

def sigmoid(result):
    """Sigmoid function."""
    return 1.0/(1.0 +np.exp(-result))

def sigmoid_prime(result):
    """Take derivative of the sigmoid function."""
    # Pretty amazing math trick here from notes.
    return sigmoid(result)*(1-sigmoid(result))


network = PerceptronManager()
network.menu()
