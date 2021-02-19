#!/usr/bin/env python

import argparse
import itertools
import numpy as np
import rospy

from typing import Callable, Union

from bayesopt4ros.srv import BayesOptSrv


def forrester_function(x: Union[np.ndarray, float]) -> np.ndarray:
    """! The Forrester test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html

    @param x    Input to the function.

    @return Function value at given inputs.
    """
    x = np.array(x)
    return -1 * ((6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)).squeeze()


class TestClient(object):
    """! A demonstration on how to use the BayesOpt service from a Python node. """

    def __init__(self, service_name: str, objective: Callable) -> None:
        """! Initializer of the client that queries the BayesOpt service.

        @param service_name     Name of the service (needs to be consistent with service node).
        @param objective        Name of the example objective.
        """
        rospy.init_node(self.__class__.__name__, anonymous=True, log_level=rospy.INFO)
        self.service_name = service_name
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
            print(f"Service call failed: {e}")

    def run(self) -> None:
        """! Method that starts emulates client behavior."""
        x_new = self.request(0.0)  # First value is just to trigger the service
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(f"[Client] x_new = [{p_string}]")
            y_new = self.func(x_new)
            rospy.loginfo(f"[Client] y_new = {y_new:.2f}")

            x_new = self.request(y_new)
            # NOTE(lukasfro): catching rospy.ServiceException does not seem to work here.
            if x_new is None:
                rospy.logwarn("[Client] Invalid response. Shutting down!")
                rospy.signal_shutdown("Invalid response from BayesOptService.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        help="Name of the objective function",
        type=str,
        choices=["Forrester"],
    )
    args = parser.parse_args()
    try:
        node = TestClient(service_name="BayesOpt", objective=args.objective)
        node.run()
    except rospy.ROSInterruptException:
        pass