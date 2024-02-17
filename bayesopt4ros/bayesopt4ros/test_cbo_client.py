#!/usr/bin/env python3


import itertools
import unittest
import numpy as np
import rclpy
import pytest
import torch
import threading
import time

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

        self.test_point_id = 0
        # self.event = threading.Event()


            
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
            # self.get_logger().info(f"dbg request_param: goal = {goal}")
            self._send_goal_future = self.client.send_goal_async(goal,
                feedback_callback=None)
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
        self.get_logger().info(f"[Client] c_prev = [{self.c_prev}]")
        
        # Emulate experiment by querying the objective function
        xc_new = torch.atleast_2d(torch.cat((torch.tensor(x_new,dtype=torch.double), self.c_prev)))
        y_new = self.func(xc_new).squeeze().item()
        self.get_logger().info(f"[Client] y_new = {y_new:.2f}")
        # Request server and obtain new parameters
        if self.iter < self.iterMax:
            c_new = self.sample_context()
            self.request_parameter(y_new, c_new = c_new) #TODO: does it assume y_new belongs to this c_new or previous one?!
            self.c_prev = c_new
        else: #end of training
            self.start_testing()


    def request_bayesopt_state(self, context):
        """Method that requests the (final) state of BayesOpt server.

        .. note:: As we only call this function once, we can just create the
            corresponding client locally.
        """
        state_client = ActionClient(self, ContextualBayesOptState, "ContextualBayesOptState")
        state_client.wait_for_server()
        goal = ContextualBayesOptState.Goal()
        goal.context = context #.tolist() ?
        self.state_send_goal_future = state_client.send_goal_async(goal)
        self.get_logger().info(f"sent goal with context = {context} to get the final results!")
        self.state_send_goal_future.add_done_callback(self.state_response_callback)

    def state_response_callback(self,future):
        """A callback to know if the server rejected the goal immediately or accepted and will be treated"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected by the server:( !!!!!!!!!!!! I dont know why! where x_new={goal_handle.request}')
            return
        self.get_logger().info('Goal accepted to be treated :)')
        self._get_result_state_future = goal_handle.get_result_async()
        self._get_result_state_future.add_done_callback(self.get_result_state_callback)

    def get_result_state_callback(self, future):
        result = future.result().result
        # Save results
        self.results.append(result)
        # self.results[self.test_point_id].x_opt = result.x_opt
        # self.results[self.test_point_id].f_opt = result.f_opt
        # trigger next request
        self.test_point_id +=1
        if self.test_point_id < self.num_test_points:
            self.request_bayesopt_state(context=self.func._test_contexts[self.test_point_id])
        else:
            self.check_results()

        
    def run(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server
        c_new = self.sample_context()
        self.c_prev = c_new
        self.request_parameter(y_new=0.0, c_new=c_new) #x_new = 

    def sample_context(self) -> np.ndarray:
        """Samples a random context variable to emulate the client."""
        context_bounds = [b for b in self.func._bounds[self.func.input_dim :]]
        context = torch.tensor([np.random.uniform(b[0], b[1]) for b in context_bounds])  
        return context
    
    def start_testing(self) -> None:
        """Testing the client on the defined objective function and couple of
        contexts."""
        self.results = []
        self.num_test_points = len(self.func._test_contexts)
        self.request_bayesopt_state(context=self.func._test_contexts[0])

    def check_results(self):
        id = 0
        # Check the estimated optimum for different contexts
        for context, x_opt, f_opt in zip(
            self.func._test_contexts,
            self.func._contextual_optimizer,
            self.func._contextual_optimal_values,
        ):
            result = self.results[id]
            # self.get_logger().info(f"dbg- check_results: result is following {result}")
            # formatted_values = [f"{value:.3f}" for value in result.x_opt]
            formatted_x_opt = ', '.join([f'{x:.3f}' for x in result.x_opt])
            self.get_logger().info(f"[{id}]- result.x_opt = {formatted_x_opt},  x_opt = {x_opt}")
            self.get_logger().info(f"[{id}]- result.f_opt = {result.f_opt:.3f},  f_opt = {f_opt}")

            # Get the (estimated) optimum of the objective for a given context
            # Be kind w.r.t. precision of solution
            np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=1)
            np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=1)
            id +=1
        self.get_logger().warn("--------------------------------------------------------")
        self.get_logger().warn("Congrats! All test has passed successfully with contextual_forrester function!")
        rclpy.shutdown()



def main(args=None):
    # rclpy.init(args=args)

    test_client = ExampleContextualClient(
            server_name="ContextualBayesOpt",
            objective="ContextualForrester",
            maximize=False,
        )
    test_client.run()
    # rclpy.logging.get_logger("test_CBO_client").warn(f"Objective: {objective}")
    rclpy.spin(test_client)

if __name__ == "__main__":
    main()



# def LOGIC_OF_RUNNING(self) -> None:
#     """Method that emulates client behavior."""
#     # First value is just to trigger the server

#     c_new = self.sample_context()
#     x_new = self.request_parameter(y_new=0.0, c_new=c_new)

#     # Start querying the BayesOpt server until it reached max iterations
#     for iter in itertools.count():
#         xc_new = torch.atleast_2d(torch.cat((x_new, c_new)))
#         y_new = self.func(xc_new).squeeze().item()
#         c_new = self.sample_context()
#         x_new = self.request_parameter(y_new=y_new, c_new=c_new)
