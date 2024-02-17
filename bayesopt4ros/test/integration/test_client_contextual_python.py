#!/usr/bin/env python3


import itertools
import unittest
import numpy as np
import rclpy
import pytest
import torch

from rclpy.action import ActionClient
from rclpy.node import Node

from typing import Callable

from bayesopt4ros import test_objectives
from bayesopt_actions.action import ContextualBayesOpt, ContextualBayesOptState



class ExampleContextualClient(Node):
    """A demonstration on how to use the contexutal BayesOpt server from a Python
    node."""

    def __init__(self, server_name: str, objective: Callable, maximize=True) -> None:
        """Initializer of the client that queries the contextual BayesOpt server.

        Parameters
        ----------
        server_name : str
            Name of the server (needs to be consistent with server node).
        objective : str
            Name of the example objective.
        maximize : bool
            If True, consider the problem a maximization problem.
        """
        rclpy.init()
        super().__init__('test_ContextBO_node')        

        self.client = ActionClient(self, ContextualBayesOpt, server_name)
        self.iter = 0
        self.iterMax = 30 #
        self.client.wait_for_server()

        if objective == "ContextualForrester":
            self.func = test_objectives.ContextualForrester()
        else:
            raise ValueError("No such objective.")

        self.maximize = maximize
        self.y_best = -np.inf if maximize else np.inf
        self.x_best = None

    def request_parameter(self, y_new: float, c_new: np.ndarray): # -> np.ndarray:
        """Method that requests new parameters from the ContextualBayesOpt
        server for a given context.

        Parameters
        ----------
        y_new : float
            The function value obtained from the objective/experiment.
        c_new : np.ndarray
            The context variable for the next evaluation/experiment.

        Returns
        -------
        numpy.ndarray
            An array containing the new parameters suggested by contextual BayesOpt
            server.
        """
        self.iter += 1
        goal = ContextualBayesOpt.Goal(y_new=y_new, c_new=c_new)
        self._send_goal_future = self.client.send_goal_async(goal,feedback_callback=None)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
            """A callback to know if the server rejected the goal immediately or accepted and will be treated"""
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info(f'Goal rejected by the server:( !!!!!!!!!!!! I dont know why! where x_new={goal_handle.request}')
                return
            self.get_logger().info('Goal accepted to be treated :)')
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):

        result = future.result().result
        x_new = result.x_new
        if not len(x_new):
            self.get_logger().info("[Client] Terminating - invalid response from server.")
        self.get_logger().info(f"[Client] Iteration {self.iter + 1}")
        p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
        self.get_logger().info(f"[Client] x_new = [{p_string}]")
        c_new = self.sample_context()
        # Emulate experiment by querying the objective function
        xc_new = torch.atleast_2d(torch.cat((torch.tensor(x_new,dtype=torch.double), c_new)))
        y_new = self.func(xc_new).squeeze().item()
        self.get_logger().info(f"[Client] y_new = {y_new:.2f}")
        # Request server and obtain new parameters
        if self.iter < self.iterMax:
            self.request_parameter(y_new, c_new = c_new) #TODO: does it assume y_new belongs to this c_new or previous one?!
        # else:
            # result = self.request_bayesopt_state(context=[0.7])

    def request_bayesopt_state(self, context) -> ContextualBayesOptState.Result:
        """Method that requests the (final) state of BayesOpt server.

        .. note:: As we only call this function once, we can just create the
            corresponding client locally.
        """
        state_client = ActionClient(self, ContextualBayesOptState, "ContextualBayesOptState")
        state_client.wait_for_server()

        goal = ContextualBayesOptState.Goal()
        goal.context = context #.tolist() ?
        state_client.send_goal(goal)
        state_client.wait_for_result()
        return state_client.get_result()

    def run(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server

        c_new = self.sample_context()
        x_new = self.request_parameter(y_new=0.0, c_new=c_new)

        # Start querying the BayesOpt server until it reached max iterations
        for iter in itertools.count():
            self.get_logger().info(f"[Client] Iteration {iter + 1}")
            x_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            c_string = ", ".join([f"{xi:.3f}" for xi in c_new])
            self.get_logger().info(f"[Client] x_new = [{x_string}] for c_new = [{c_string}]")

            # Emulate experiment by querying the objective function
            xc_new = torch.atleast_2d(torch.cat((x_new, c_new)))
            y_new = self.func(xc_new).squeeze().item()

            if (self.maximize and y_new > self.y_best) or (
                not self.maximize and y_new < self.y_best
            ):
                self.y_best = y_new
                self.x_best = x_new

            self.get_logger().info(f"[Client] y_new = {y_new:.2f}")

            # Request server and obtain new parameters
            c_new = self.sample_context()
            x_new = self.request_parameter(y_new=y_new, c_new=c_new)
            if not len(x_new):
                self.get_logger().info("[Client] Terminating - invalid response from server.")
                break

    def sample_context(self) -> np.ndarray:
        """Samples a random context variable to emulate the client."""
        context_bounds = [b for b in self.func._bounds[self.func.input_dim :]]
        context = torch.tensor([np.random.uniform(b[0], b[1]) for b in context_bounds])
        return context


class ContextualClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary contextual Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective function and couple of
        contexts."""

        # Set up the client
        node = ExampleContextualClient(
            server_name="ContextualBayesOpt",
            objective=self._objective_name,
            maximize=self._maximize,
        )

        # Emulate experiment
        node.run()

        # Check the estimated optimum for different contexts
        for context, x_opt, f_opt in zip(
            node.func._test_contexts,
            node.func._contextual_optimizer,
            node.func._contextual_optimal_values,
        ):
            # Get the (estimated) optimum of the objective for a given context
            result = node.request_bayesopt_state(context)

            # Be kind w.r.t. precision of solution
            np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=1)
            np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=1)


class ContextualClientTestCaseForrester(ContextualClientTestCase):
    _objective_name = "ContextualForrester"
    _maximize = False





def main(args=None):
    # rclpy.init(args=args)
    objective = "ContextualForrester" #old:rospy.get_param("/objective")
    # rclpy.logging.get_logger("test_CBO_client").warn(f"Objective: {objective}")
    cbo_client_case = ContextualClientTestCaseForrester()

    # if objective == "ContextualForrester":
    #     rostest.rosrun(
    #         "bayesopt4ros", "test_python_client", ContextualClientTestCaseForrester
    #     )
    # else:
    #     raise ValueError("Not a known objective function.")

if __name__ == "__main__":
    # Note: unfortunately, rostest.rosrun does not allow to parse arguments
    # This can probably be done more efficiently but honestly, the ROS documentation for
    # integration testing is kind of outdated and not very thorough...
    main()
