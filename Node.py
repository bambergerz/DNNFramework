import numpy as np
from abc import abstractmethod
import matplotlib as plt


class Node:
    """
    Nodes are meant to simulate a particular function
    (e.g., or Relu or Softmax).

    This type of node accepts a single input and produces a single output

    This nodes has Only Forward and Backwards methods which are
    implemented by its children classes.
    """

    def __init__(self):
        """
        Saves the values from the forward pass for
         the backward pass soon to come.
        """
        self.value = None

    @abstractmethod
    def forward(self, forward_in):
        """

        :param forward_in: The input to this node
        :return: The output of this node.

        Side effect: set self.value to the input from
        the forward pass
        """
        self.value = forward_in

    @abstractmethod
    def backward(self, back_in):
        pass



class Relu(Node):
    """
    Represents a Relu Node instance.

    Forward input is an n x m matrix
    (m samples of n dimensions)
    Forward output is n x m matrix

    Backward input is an n' x m' matrix.
    (m' samples of n' dimensions)
    Backward output is also an n' x m' matrix.
    """

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, forward_in):
        """

        :param forward_in: n x m maxtrix
        :return: n x m matrix
        """
        super(Relu, self).forward(forward_in)
        return np.maximum(forward_in, 0)

    def backward(self, back_in):
        """

        :param back_in: n x m matrix representing gradient
        of m different samples.
        :return: n x m matrix representing the gradient
        of the Relu functtion. negative entries have
        gradient of 0 while non-negative entries have a gradient of 1.
        """
        self.value[self.value >= 0] = 1
        self.value[self.value < 0] = 0
        return self.value * back_in


class Sigmoid(Node):
    """
    Represents a Relu Node instance.

    f(x) = (e^x) / (e^x + 1)
    f'(x) = f(x) * (1 - f(x))

    https://en.wikipedia.org/wiki/Sigmoid_function

    Forward input is an n x m matrix
    (m samples of n dimensions)
    Forward output is n x m matrix

    Backward input is an n' x m' matrix.
    (m' samples of n' dimensions)
    Backward output is also an n' x m' matrix.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, forward_in):
        super(Sigmoid, self).forward(forward_in)
        return 1 / (1 + np.exp(-1 * forward_in))

    def backward(self, back_in):
        grad = self.forward(self.value) * (1 - self.forward(self.value))
        return back_in * grad


class Softmax(Node):
    """
    Represents a Softmax Node instance.

    f(x) = (e^x) / (sum_{i=0}^{n} e ^ x[i])

    https://en.wikipedia.org/wiki/Sigmoid_function

    Forward input is an n x m matrix
    (m samples of n dimensions)
    Forward output is n x m matrix

    Backward input is an n' x m' matrix.
    (m' samples of n' dimensions)
    Backward output is also an n' x m' matrix.
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, forward_in):
        super(Softmax, self).forward(forward_in)
        return np.exp(forward_in) / np.sum(np.exp(forward_in), axis=0)

    def backward(self, back_in):
        s = self.forward(self.value).T
        grad = back_in.T
        out = np.zeros(self.value.shape.T)
        for index, sample in enumerate(s):
            jac = self.jacobian(sample)
            out[index] = grad[index] @ jac
        return out.T

    def jacobian(self, s):
        return np.diag(s) - np.outer(s, s)


class Gate:
    """
    Gates are meant to simulate a particular function
    (e.g., addition or multiplication.

    This type of node accepts a single input and produces a single output

    This nodes has Only Forward and Backwards methods which are
    implemented by its children classes.
    """

    def __init__(self):
        self.value = (None, None)

    @abstractmethod
    def forward(self, forward_in_a, forward_in_b):
        self.value = (forward_in_a, forward_in_b)

    @ abstractmethod()
    def backward(self, back_in):
        pass


class Times(Gate):
    """
    Represents a Multiplication Gate instance.

    f(a,b) = a * b
    grad(f) wrt. a = b
    grad(f) wrt. b = a

    Multiplication gates are usually used
    in the beggining of a layer to multiply the
    output of the previous layer with the weights
    of the current one.

    The first input, a, represents the output from
    the previous layer. This is an n x m matrix.
    m samples of n dimensions.

    The second input, b, represents the weights of
    the weights for this node. This is an k x n matrix.

    forward output is a k x m matrix. I.e., m samples of k
    dimensions. This is how we change the dimensions between
    consecutive layers.

    Backward input is a k x n matrix representing the
    current gradient.
    (n samples of k dimensions)

    Backward output are two matrices
    the first is an n x n matrix representing the gradient of x
    the second is a k x n matrix representing the gradient of w
    """

    def __init__(self):
        super(Times, self).__init__()

    def forward(self, forward_in_a, forward_in_b):
        """
        Given inputs a and b return np.multiply(b,a)
        i.e., we perform the matrix multiplication of b and a

        :param forward_in_a: n x m matrix representing the input x
        :param forward_in_b: k x n matrix representing the input w
        :return: k x m matrix representing output of the multiplication.
        """
        super(Times, self).forward(forward_in_a, forward_in_b)
        return forward_in_b @ forward_in_a

    def backward(self, back_in):
        """

        Consider the gradient of this multiplication
        function with respect to both inputs.

        grad_x_dims = (n x k) * (k x n) ==> (n x n)
        grad_w_dims = (k x n) * (n x k) ==> (k x k)

        :param back_in: a matrix of dimensions k x m
        :return: grad_x, a matrix of dimensions n x m
                 grad_w, a matrix of dimensions k x n
        """
        grad_x = self.value[1].T @ back_in
        grad_w = back_in @ self.value[0].T
        return grad_w, grad_x


class Plus(Gate):
    def __init__(self):
        super(Plus, self).__init__()

    def forward(self, forward_in_a, forward_in_b):
        super(Plus, self).forward(forward_in_a, forward_in_b)
        return forward_in_a + forward_in_a

    def backward(self, back_in):
        return back_in, back_in




