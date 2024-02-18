#!/usr/bin/env python3


import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import numpy as np
import itertools
import torch
from typing import Callable

# import actionlib
# import itertools
# import unittest
# import rostest

from typing import Callable

from bayesopt4ros import test_objectives
from bayesopt_actions.action import ContextualBayesOpt, ContextualBayesOptState

from line_profiler import LineProfiler

class ContextualBayesOptClient(Node):
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
        super().__init__('contextual_bayesopt_action_client_node')        
        self.client = ActionClient(self, ContextualBayesOpt, server_name)
        self.iter = 0
        self.iterMax = 17 #should be bigger that n_init+1 in server side, TODO: make it using itertools as in the server and to be from the config file (or maybe it's not related?!)

        self.client.wait_for_server()

        if objective == "myFun": #"ContextualForrester":
            self.func = lambda x: x[:, 0] + x[:, 1] + 3*torch.sin(x[:,0]+x[:,2]) #test_objectives.ContextualForrester()
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

        # self.get_logger().info(f"dbg request_param: goal = {goal}")
        self._send_goal_future = self.client.send_goal_async(goal,
            feedback_callback=None)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        # self.client.send_goal(goal)
        # self.client.wait_for_result()
        # result = self.client.get_result()
        # if(result == None):
        #     return torch.tensor([0.0])
        # else:
        #     return torch.tensor(result.x_new)

    def goal_response_callback(self, future):
            """A callback to know if the server rejected the goal immediately or accepted and will be treated"""
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info(f'Goal rejected by the server:( !!!!!!!!!!!! I dont know why! where x_new={goal_handle.request}')
                return
            self.get_logger().info('Goal accepted to be treated :)')
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
            # self.get_logger().info(f"dbg get_response_cb: result = {goal_handle}")

    def get_result_callback(self, future):
        # self.get_logger().info(f"dbg get_result_cb: begining")
        result = future.result().result
        x_new = result.x_new
        if not len(x_new):
            self.get_logger().info("[Client] Terminating - invalid response from server.")
        # self.get_logger().info('Result: {0}'.format(result.x_new))
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
        else:
            result = self.request_bayesopt_state(context=[0.7])  #for testing


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
        # x_new = result.x_new
        # True optimum of the objective
        x_opt = result.x_opt
        f_opt = result.f_opt

        self.get_logger().info("[Client] for context = 0.7")
        self.get_logger().info(f"[Client] x_opt {x_opt}, x_true_opt = (-2.61, -5.0)")
        self.get_logger().info(f"[Client] f_opt {f_opt}, f_true_opt = -10.4388")

        # self.request_bayesopt_state(context=0.1)
        # x_opt = result.x_opt
        # f_opt = result.f_opt
        # self.get_logger().info("[Client] for context = 0.1")
        # self.get_logger().info(f"[Client] x_opt {x_opt},  x_true_opt = (-2.011, -5.0)")
        # self.get_logger().info(f"[Client] f_opt {f_opt}, f_true_opt = -9.839")


    # def run(self) -> None:
    #     """Method that emulates client behavior."""
    #     # First value is just to trigger the server

    #     c_new = self.sample_context()
    #     # print(f"c_new0 ={c_new}\n")
    #     x_new = self.request_parameter(y_new=0.0, c_new=c_new)

    #     # Start querying the BayesOpt server until it reached max iterations
    #     for iter in itertools.count():
    #         rospy.loginfo(f"[Client] Iteration {iter + 1}")
    #         x_string = ", ".join([f"{xi:.3f}" for xi in x_new])
    #         c_string = ", ".join([f"{xi:.3f}" for xi in c_new])
    #         rospy.loginfo(f"[Client] x_new = [{x_string}] for c_new = [{c_string}]")

    #         # Emulate experiment by querying the objective function
    #         xc_new = torch.atleast_2d(torch.cat((x_new, c_new)))
    #         print(f"xc_new={xc_new}\n")
    #         y_new = self.func(xc_new).squeeze().item()

    #         if (self.maximize and y_new > self.y_best) or (
    #             not self.maximize and y_new < self.y_best
    #         ):
    #             self.y_best = y_new
    #             self.x_best = x_new

    #         rospy.loginfo(f"[Client] y_new = {y_new:.2f}")

    #         # Request server and obtain new parameters
    #         c_new = self.sample_context()
    #         x_new = self.request_parameter(y_new=y_new, c_new=c_new)
    #         if not len(x_new):
    #             rospy.loginfo("[Client] Terminating - invalid response from server.")
    #             break

    def sample_context(self) -> np.ndarray:
        """Samples a random context variable to emulate the client."""
        # context_bounds = [b for b in self.func._bounds[self.func.input_dim :]]
        context_bounds = [[0,1]]
        context = torch.tensor([np.random.uniform(b[0], b[1]) for b in context_bounds],dtype=torch.double)
        return context

    def start_sending(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server
        self.c_prev = self.sample_context()
        self.request_parameter(y_new=0.0, c_new=self.c_prev)


# class ContextualClientTestCase(unittest.TestCase):
#     """Integration test cases for exemplary contextual Python client."""

#     _objective_name = None
#     _maximize = True

#     def test_objective(self) -> None:
#         """Testing the client on the defined objective function and couple of
#         contexts."""

#         # Set up the client
#         node = ExampleContextualClient(
#             server_name="ContextualBayesOpt",
#             objective=self._objective_name,
#             maximize=self._maximize,
#         )

#         # Emulate experiment
#         node.run()

#         # Check the estimated optimum for different contexts
#         for context, x_opt, f_opt in zip(
#             node.func._test_contexts,
#             node.func._contextual_optimizer,
#             node.func._contextual_optimal_values,
#         ):
#             # Get the (estimated) optimum of the objective for a given context
#             result = node.request_bayesopt_state(context)

#             # Be kind w.r.t. precision of solution
#             np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=1)
#             np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=1)


# class ContextualClientTestCaseForrester(ContextualClientTestCase):
#     _objective_name = "ContextualForrester"
#     _maximize = False


# if __name__ == "__main__":
#     # Note: unfortunately, rostest.rosrun does not allow to parse arguments
#     # This can probably be done more efficiently but honestly, the ROS documentation for
#     # integration testing is kind of outdated and not very thorough...
#     objective = rospy.get_param("/objective")
#     rospy.logwarn(f"Objective: {objective}")
#     if objective == "ContextualForrester":
#         rostest.rosrun(
#             "bayesopt4ros", "test_python_client", ContextualClientTestCaseForrester
#         )
#     else:
#         raise ValueError("Not a known objective function.")

def main(args=None):
    rclpy.init(args=args)

    action_client = ContextualBayesOptClient(server_name='ContextualBayesOpt', objective="myFun", maximize=False)
    action_client.start_sending()
    rclpy.spin(action_client)
    rclpy.shutdown()


if __name__ == '__main__':
    main()