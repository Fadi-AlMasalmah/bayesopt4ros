
# !/usr/bin/env python3
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
# limitations under the License.import time


import time


import rclpy
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from bayesopt4ros.bayesopt import BayesianOptimization
from bayesopt4ros import util
from bayesopt_actions.action import BayesOpt, BayesOptState
# from bayesopt4ros.msg import BayesOptResult, BayesOptAction
# from bayesopt4ros.msg import BayesOptStateResult, BayesOptStateAction


class BayesOptServer(Node):

    def __init__(
        self,
        # config_file: str, # we get this as a ros_param inside the class
        server_name: str = "BayesOpt",
        log_file: str = None,
        # anonymous: bool = True,
        # log_level: int = rclpy.INFO,
        silent: bool = False,
        # node_rate: float = 5.0,
    ):
        super().__init__(node_name='BayesOptServer_Node')
        
        self.declare_parameter('config_file', ' ')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        self._initialize_bayesopt(config_file)
        self._initialize_parameter_server(server_name)
        self._initialize_state_server(server_name + "State")

        self.request_count = 0
        self.log_file = log_file
        self.config_file = config_file
        self.silent = silent
        # self.rosrate = rospy.Rate(node_rate)
        self.get_logger().info(self._log_prefix + "Ready to receive requests.")
    
    def _initialize_bayesopt(self, config_file):
        # try:
        self.bo = BayesianOptimization.from_file(config_file,logger=self.get_logger())
        # except Exception as e:
        #     self.get_logger().error(f"[BayesOpt] Something went wrong with initialization: '{e}'")
        #     raise SystemExit 
            # rclpy.signal_shutdown("Initialization of BayesOpt failed.")

    def _initialize_parameter_server(self, server_name):
        """This server obtains new function values and provides new parameters."""
        self.parameter_server = ActionServer(
            self,
            BayesOpt,
            server_name,
            self.next_parameter_callback,
            )

    def _initialize_state_server(self, server_name):
        """This server provides the current state/results of BO."""
        self.parameter_server = ActionServer(
            self,
            BayesOptState,
            server_name,
            self.state_callback,
            )
        
    @util.count_requests
    def next_parameter_callback(self, goal_handle):
        """Method that gets called when a new parameter vector is requested.

        The action message (goal/result/feedback) is defined here:
        ``action/BayesOpt.action``

        .. literalinclude:: ../action/BayesOpt.action

        Parameters
        ----------
        goal : BayesOptAction
            The action (goal) coming from a client.
        """
        goal = goal_handle.request
        # self.get_logger().info(f'the requested goal is {goal}')
        self._print_goal(goal) if not self.silent else None
        if self._check_final_iter(goal):
            return  # Do not continue once we reached maximum iterations

        # Obtain the new parameter values.
        result = BayesOpt.Result()
        # rclpy.logging.get_logger('dbg').info(f"dbg logging_getlogger")

        x_next_t = self.bo.next(goal)

        # self.get_logger().info(f"dbg bayes_server inside next_param_callback(), x_next_t = {x_next_t}")

        result.x_new = x_next_t.tolist()

        # self.get_logger().info(f"dbg bayes_server inside next_param_callback(), after result = {result}")

        # self.parameter_server.set_succeeded(result) #maybe remove this?
        goal_handle.succeed() #TODO: check if this should .accept instead?!
        self._print_result(result) if not self.silent else None
        return result
    
    def state_callback(self, goal_handle):
        """Method that gets called when the BayesOpt state is requested.

        .. note:: We are calling this `state` instead of `result` to avoid
            confusion with the `result` variable in the action message.

        The action message (goal/result/feedback) is defined here:
        ``action/BayesOptState.action``

        .. literalinclude:: ../action/BayesOptState.action

        Parameters
        ----------
        goal : BayesOptStateAction
            The action (goal) coming from a client.
        """
        state = BayesOptState.Result()

        # Best observed variables
        x_best, y_best = self.bo.get_best_observation()
        state.x_best = x_best.tolist()
        
        state.y_best = float(y_best.item())

        # Posterior mean optimum
        x_opt, f_opt = self.bo.get_optimal_parameters()
        state.x_opt = x_opt.tolist()
        state.f_opt = f_opt.item()

        self.get_logger().info(f"[Client] x_best observed: {x_best}")
        self.get_logger().info(f"[Client] y_best observed: {y_best}") 
        self.get_logger().info(f"[Client] x_opt estimated: {x_opt}")
        self.get_logger().info(f"[Client] f_opt estimated: {f_opt}")
        goal_handle.succeed() #old: succeed(state)
        return state
      
    # def execute_callback(self, goal_handle):
    #     self.get_logger().info('Executing goal...')

    #     feedback_msg = BayesOpt.Feedback()
    #     feedback_msg.partial_sequence = [0, 1]

    #     for i in range(1, goal_handle.request.order):
    #         feedback_msg.partial_sequence.append(
    #             feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])
    #         self.get_logger().info('Feedback: {0}'.format(feedback_msg.partial_sequence))
    #         goal_handle.publish_feedback(feedback_msg)
    #         time.sleep(1)

    #     goal_handle.succeed()

    #     result = BayesOpt.Result()
    #     result.sequence = feedback_msg.partial_sequence
    #     return result
    
    def _check_final_iter(self, goal):
        if self.bo.max_iter and self.request_count > self.bo.max_iter:
            # Updates model with last function and logs the final GP model
            self.get_logger().warning("[BayesOpt] Max iter reached. No longer responding!")
            self.bo.update_last_goal(goal)
            #old: self.parameter_server.set_aborted()
            # self.parameter_server.destroy()
            return True
        else:
            return False

    def _print_goal(self, goal):
        if not self.request_count == 1:
            self.get_logger().info(self._log_prefix + f"New value: {goal.y_new:.3f}")
        else:
            self.get_logger().info(self._log_prefix + f"Discard value: {goal.y_new:.3f}")

    def _print_result(self, result):
        s = util.iter_to_string(result.x_new, ".3f")
        self.get_logger().info(self._log_prefix + f"x_new: [{s}]")
        if self.request_count < self.bo.max_iter:
            self.get_logger().info(self._log_prefix + "Waiting for new request...")

    @property
    def _log_prefix(self) -> str:
        """Convenience property that pre-fixes the logging strings."""
        return f"[{self.__class__.__name__}] Iteration {self.request_count}: "

    # @staticmethod
    # def run() -> None:
    #     """Simply starts the server."""
    #     rospy.spin()



def main(args=None):
    rclpy.init(args=args)

    # try:
    #     # config_name = [p for p in rclpy.get_param_names() if "bayesopt_config" in p]
    #     # config_file = rclpy.get_param(config_name[0])
    #     node = BayesOptServer(config_file=config_file)
    #     node.run()
    # except KeyError:
    #     rospy.logerr("Could not find the config file.")
    # except rospy.ROSInterruptException:
    #     pass

    bayesopt_action_server = BayesOptServer()

    try:
        rclpy.spin(bayesopt_action_server)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()