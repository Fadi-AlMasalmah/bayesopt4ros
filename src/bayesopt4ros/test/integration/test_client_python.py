import unittest
import itertools
import numpy as np
import rospy
import rostest

from typing import Callable, Union

from bayesopt4ros.srv import BayesOptSrv


def forrester_function(x: Union[np.ndarray, float]) -> np.ndarray:
    """! The Forrester test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html

    Note: We multiply by -1 to maximize the function instead of minimizing.

    @param x    Input to the function.

    @return Function value at given inputs.
    """
    x = np.array(x)
    return -1 * ((6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)).squeeze()


class ExampleClient(object):
    """! A demonstration on how to use the BayesOpt service from a Python node. """

    def __init__(self, service_name: str, objective: Callable) -> None:
        """! Initializer of the client that queries the BayesOpt service.

        @param service_name     Name of the service (needs to be consistent with service node).
        @param objective        Name of the example objective.
        """
        rospy.init_node(self.__class__.__name__, anonymous=True, log_level=rospy.INFO)
        self.service_name = service_name
        self.y_best, self.x_best = -np.inf, None
        if objective == "Forrester":
            self.func = forrester_function
        else:
            raise ValueError("No such objective.")

    def request(self, value: float) -> np.ndarray:
        """! Method that handles interaction with the service node.

        @param value    The function value obtained from the objective/experiment.

        @return An array containing the new parameters suggested by BayesOpt service.
        """
        rospy.wait_for_service(self.service_name)
        try:
            bayesopt_request = rospy.ServiceProxy(self.service_name, BayesOptSrv)
            response = bayesopt_request(value)
            return response.next
        except rospy.ServiceException as e:
            rospy.logwarn("[Client] Invalid response. Shutting down!")
            rospy.signal_shutdown("Invalid response from BayesOptService.")

    def run(self) -> None:
        """! Method that emulates client behavior."""
        # First value is just to trigger the service
        x_new = self.request(0.0)

        # Start querying the BayesOpt service until it reached max iterations
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(f"[Client] x_new = [{p_string}]")

            # Emulate experiment by querying the objective function
            y_new = self.func(x_new)
            if y_new > self.y_best:
                self.y_best = y_new
                self.x_best = x_new
            rospy.loginfo(f"[Client] y_new = {y_new:.2f}, y_best = {self.y_best:.2f}")

            # Request service and obtain new parameters
            x_new = self.request(y_new)
            if x_new is None:
                break


class ClientTestCase(unittest.TestCase):
    """! Integration test cases for the exemplary Python client. """

    def test_forrester(self) -> None:
        """! Testing client on 1-dimensional Forrester function."""
        node = ExampleClient(service_name="BayesOpt", objective="Forrester")
        node.run()

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(node.x_best, np.array([0.757]), decimal=3)
        np.testing.assert_almost_equal(node.y_best, np.array([6.021]), decimal=3)


if __name__ == "__main__":
    rostest.rosrun("bayesopt4ros", "test_python_client", ClientTestCase)
