# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python3


import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import numpy as np
import itertools
import torch
from typing import Callable

from bayesopt4ros import test_objectives
from bayesopt_actions.action import BayesOpt, BayesOptState


class BayesOptClient(Node):
    """A demonstration on how to use the BayesOpt server from a Python node."""

    def __init__(self, server_name: str, objective: Callable, maximize=True):
        """Initializer of the client that queries the BayesOpt server.
         Parameters
        ----------
        server_name : str
            Name of the server (needs to be consistent with server node).
        objective : str
            Name of the example objective.
        maximize : bool
            If True, consider the problem a maximization problem.
        """
        super().__init__('bayesopt_action_client_node')
        self.client = ActionClient(self, BayesOpt, server_name)
        self.iter = 1
        self.iterMax = 12 #TODO: make it using itertools as in the server and to be from the config file (or maybe it's not related?!)
        # self.client = actionlib.SimpleActionClient(server_name, BayesOptAction)
        # self.client.wait_for_server()
        if objective == "Forrester":
            self.func = test_objectives.Forrester()
        if objective == "myFun":
            self.func = lambda x : torch.square(x - 4.2)
        else:
            raise ValueError("No such objective.")
        self.maximize = maximize

    # def send_goal(self, order):
    #     goal_msg = BayesOpt.Goal()
    #     goal_msg.order = order

    #     self.client.wait_for_server()

    #     self._send_goal_future = self.client.send_goal_async(
    #         goal_msg,
    #         feedback_callback=self.feedback_callback)

    #     self._send_goal_future.add_done_callback(self.goal_response_callback)

    def request_parameter(self, y_new: float) -> np.ndarray:
        """Method that requests new parameters from the BayesOpt server.

        Parameters
        ----------
        value : float
            The function value obtained from the objective/experiment.

        Returns
        -------
        numpy.ndarray
            An array containing the new parameters suggested by BayesOpt server.
        """
        self.iter += 1
        goal = BayesOpt.Goal(y_new=y_new)
        self.client.wait_for_server() #TODO: remove if inside a realtime code?!

        # self.client.send_goal(goal)
        # self.client.wait_for_result() #TODO: make it async using futures 

        self._send_goal_future = self.client.send_goal_async(goal,
            feedback_callback=None)

        self._send_goal_future.add_done_callback(self.goal_response_callback)
        # result = self.client.get_result()
        # self.get_logger().info(f"result is {result.x_new}")
        # return 0.0 #result.x_new
    
    def goal_response_callback(self, future):
        """A callback to know if the server rejected the goal immediately or accepted and will be treated"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected by the server:( !!!!!!!!!!!! I dont know why! where x_new={goal_handle.request}')
            return
        self.get_logger().info('Goal accepted to be treated :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

        # result = self.request_bayesopt_state()
        

    def get_result_callback(self, future):
        result = future.result().result
        x_new = result.x_new

        if not len(x_new):
            self.get_logger().info("[Client] Terminating - invalid response from server.")
            
        # self.get_logger().info('Result: {0}'.format(result.x_new))
        self.get_logger().info(f"[Client] Iteration {self.iter + 1}")
        p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
        self.get_logger().info(f"[Client] x_new = [{p_string}]")

        # Emulate experiment by querying the objective function
        y_new = self.func(torch.atleast_2d(torch.tensor(x_new,dtype=torch.double))).squeeze().item()
        self.get_logger().info(f"[Client] y_new = {y_new:.2f}")

        # Request server and obtain new parameters
        if self.iter < self.iterMax:
            self.request_parameter(y_new)
        else:
            result = self.request_bayesopt_state()
           
        
    def request_bayesopt_state(self): #old: -> BayesOptState.Result:
        """Method that requests the (final) state of BayesOpt server.

        .. note:: As we only call this function once, we can just create the
            corresponding client locally.
        """
        state_client = ActionClient(self, BayesOptState, "BayesOptState")
        state_client.wait_for_server()

        goal = BayesOptState.Goal()
        self.state_send_goal_future = state_client.send_goal_async(goal)
        self.get_logger().info("sent empty goal to get the final results!")
        self.state_send_goal_future.add_done_callback(self.state_response_callback)
        # state_client.wait_for_result()
        # return result #state_client.get_result()
    
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
        self.get_logger().info(f"[Client] x_opt {x_opt.tolist()}")
        self.get_logger().info(f"[Client] f_opt {f_opt}")

        
    def start_sending(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server
        x_new = self.request_parameter(0.0)

        # # Start querying the BayesOpt server until it reached max iterations
        # for iter in itertools.count():
        #     self.get_logger().info(f"[Client] Iteration {iter + 1}")
        #     p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
        #     self.get_logger().info(f"[Client] x_new = [{p_string}]")

        #     # Emulate experiment by querying the objective function
        #     y_new = self.func(torch.atleast_2d(torch.tensor(x_new))).squeeze().item()
        #     self.get_logger().info(f"[Client] y_new = {y_new:.2f}")

        #     # Request server and obtain new parameters
        #     x_new = self.request_parameter(y_new)
        #     if not len(x_new):
        #         self.get_logger().info("[Client] Terminating - invalid response from server.")
        #         break
        # result = self.request_bayesopt_state()
        # # True optimum of the objective
        # x_opt = result.x_opt
        # f_opt = result.f_opt
        # self.get_logger().info(f"[Client] x_opt {x_opt}")
        # self.get_logger().info(f"[Client] f_opt {f_opt}")
    
    # def goal_response_callback(self, future):
    #     goal_handle = future.result()

    #     if not goal_handle.accepted:
    #         self.get_logger().info('Goal rejected :(')
    #         return

    #     self.get_logger().info('Goal accepted :)')

    #     self._get_result_future = goal_handle.get_result_async()

    #     self._get_result_future.add_done_callback(self.get_result_callback)

    # def get_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info('Result: {0}'.format(result.sequence))
    #     rclpy.shutdown()

    # def feedback_callback(self, feedback_msg):
    #     feedback = feedback_msg.feedback
    #     self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))


def main(args=None):
    rclpy.init(args=args)

    action_client = BayesOptClient(server_name='BayesOpt', objective="myFun", maximize=False)

    # action_client.request_parameter(0.0)
    action_client.start_sending()
    rclpy.spin(action_client)

    rclpy.shutdown()


if __name__ == '__main__':
    main()