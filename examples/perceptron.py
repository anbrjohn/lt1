#!/usr/bin/env python3
# By Jon Dehdari, 2016

""" Trains and runs a perceptron, graphically displaying the output.
    Documentation added by Andrew Johnson
    for Language Technology I
    14.12.2016
"""

import sys
import numpy as np # Needed for random number generation, tanh, and dot product
import matplotlib.pyplot as plt # Needed to view the results graphically

labels = [0, 1] # Represent the two groups the data belong to
#num_samples = 30
num_samples = 200
np.random.seed(seed=3) # Assigns a specific seed so the outcome isn't different each time
learning_rate = 1.0 # (See explanation under __init__ below)
test_percentage = 10
test_size = num_samples // test_percentage # In this case, save 10% of 200, or 20 samples for testing
train_size = num_samples - test_size # The remaining 180 are for training


class Perceptron(list):
    """ This class contains all the machinery to train and run a perceptron.
        The class Perceptron subclasses from type list (takes its methods)
        
    >>> fig = plt.figure()
    >>> subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5))
    >>> train_set = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    >>> test_set  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    >>> p = Perceptron(2)
    >>> p.train(train_set, test_set, status=10, subplot=subplot, fig=fig)
    """

    def __init__(self, num_inputs, learning_rate=learning_rate):
        """Args:
               num_inputs: The size of the input
               learning_rate: Aka step size. How 'smoothly' the learning occurs.
                   Smaller = 'baby steps', taking longer. Between 0 and 1.
        """
        # A specified amount of random numbers with a normal (Gaussian) distribution.
        # This is what the perceptron is trying to find a boundary for. 
        self.params = np.random.normal(size=num_inputs) 
        self.bias   = np.random.normal() # Initial bias term to offset the linear separator
        self.learning_rate = learning_rate

    def __repr__(self):
        """How the object is to be represented as a string in the interpreter"""
        return str(np.concatenate((self.params, [self.bias])))


    def error(self, guess, correct):
        """The closer this gets to zero, the better the accuracy is.

           Args:
               guess: The prediction from predict function over the training set
               correct: The actual label

           Returns:
               The number or wrong guesses.
        """
        return correct - guess


    def activate(self, val, fun='step'):
        """Selects and applies step or sigmoid function. In effect, determining
           which label a given value is grouped with (0 or 1).

            Args:
                val: Input value
                fun: Which function to use

            Returns:
                The results of applying designated function to given input
        """
        if fun == 'step':
        #Step, aka threshold function. Returns discreet outputs (1 or 0).
            if val >= 0:
                return 1
            else:
                return 0

        elif fun == 'tanh':
            # A sigmoid function. Squeezes values into between 1 and -1 and is differentiable.
            return np.tanh(val)


    def predict(self, x):
        """Takes the dot product of weights (x) and the vector and adds the bias,
           then applies the activation function to those results.
        """
        #if 0 > np.tanh(np.dot(x, self.params) + self.bias):
        return self.activate(np.dot(x, self.params) + self.bias)


    def predict_set(self, test_set):
        """Evaluates the correctness of predictions against the test/development set.

            Args:
                test_set: Set to be evaluated.

            Returns:
                A number from 1 to 0 that represents the amount of error,
                    with lower outcomes representing more error.
        """
        errors = 0
        for x, y in test_set:
            out = self.predict(x)
            if out != y: # If the prediction doesn't match what's given in test_set
                errors +=1  
        return 1 - errors / len(test_set)
        #eg: No error is 1-(0/20) = 1.0, all error is 1-(20/20) = 0.0


    def decision_boundary(self):
        """ Returns two points, along which the decision boundary for a binary classifier lies. """
        return ((0, -self.bias / self.params[1]), (-self.bias / self.params[0], 0))


    def train(self, train_set, dev_set, status=100, epochs=10, subplot=None, fig=None):
        """Trains the model over multiple epochs and displays results.

        Args:
            train_set: (x, y) tuples with respective labels to train the model
            dev_set: (x, y) tuples with respective labels to test the model
            status=100: Number of iterations. How often to draw a new line on the
                graph and print a status update.
            epochs=10: How many times the perceptron cycles over all training data

        Returns:
            (Doesn't return a value, but visually displays results)
        """
        print("Starting dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
        fig.canvas.draw()

        iterations = 0
        for epoch in range(epochs):
            np.random.shuffle(train_set)  # To ensure training/weight adjustment isn't affected by order
            for x, y in train_set:
                iterations += 1  
                out = self.predict(x)
                error = self.error(out, y)
                if error != 0:
                    #print("out=", out, "; y=", y, "; error=", error, file=sys.stderr)
                    self.bias += self.learning_rate * error # Uses error to readjust bias so it gradually reflects the actual data.
                    for i in range(len(x)):
                        self.params[i] += self.learning_rate * error * x[i]  # Uses error to readjust weights.
                if iterations % status == 0:
                    print("Dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
                    subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
                    fig.canvas.draw()
            self.learning_rate *= 0.9  # Incrementally decreases the learning rate
            # on the thinking that larger 'jumps' are appropriate initially,
            # while 'smaller steps' are needed for finding precisely the right maixmum.
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], 'g-x', linewidth=5)
        fig.canvas.draw()


def main(): 
    import doctest
    doctest.testmod()  # Tests the code based on looking at docrings after '>>>'
    # Only returns something if test fails (unless add -v for verbose to command)

    # Generates two groups of data points, which this perceptron tries to delineate
    # Blue and red have distinct mean locations (loc)
    # The standard deviation (scale) for both is narrower along the y-axis
    x_blue = np.random.normal(loc=0, size=num_samples)
    y_blue = np.random.normal(loc=0, scale=0.5, size=num_samples) 
    x_red  = np.random.normal(loc=2, size=num_samples)
    y_red  = np.random.normal(loc=2, scale=0.5, size=num_samples)
    
    data =  list(zip(zip(x_blue, y_blue), [labels[0]] * num_samples)) # Pair of random (x, y) points and the label 0
    data += zip(zip(x_red,  y_red),  [labels[1]] * num_samples) # Same but with label 1
    np.random.shuffle(data)  # Shuffles order so training/test sets will include items from both groups
    
    train_set = data[:train_size]
    test_set  = data[train_size:]
    
    # Matplotlib craziness to be able to update a plot
    plt.ion()
    fig = plt.figure()
    subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5)) # How far the axes of the displayed graph extend
    subplot.grid()
    subplot.plot(x_blue, y_blue, linestyle='None', marker='o', color='blue')
    subplot.plot(x_red,  y_red,  linestyle='None', marker='o', color='red')

    p = Perceptron(2) #2 because there are two possible labels
    p.train(train_set, test_set, subplot=subplot, fig=fig)


# Executes the function main() if this file is being run as the main program,
# thereby preventing it from running when simply imported elsewhere as a module.
if __name__ == '__main__':
    main()
