
import rclpy
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from bayesopt4ros.contextual_bayesopt import ContextualBayesianOptimization
from bayesopt4ros import util, BayesOptServer
from bayesopt_actions.action import ContextualBayesOpt, ContextualBayesOptState

import timeit
import torch
class ContextualBayesOptServer(BayesOptServer):
    """The contextual Bayesian optimization server node.

    Acts as a layer between the actual contextual Bayesian optimization and ROS.
    """

    def __init__(
        self,
        # config_file: str, # we get this as a parameter inside the class from the laucnh file
        server_name: str = "ContextualBayesOpt",
        log_file: str = None,
        # anonymous: bool = True,
        # log_level: int = rospy.INFO,
        silent: bool = False,
        # node_rate: float = 5.0,
    ) -> None:
        """The ContextualBayesOptServer class initializer.

        For paramters see :class:`bayesopt_server.BayesOptServer`.
        """
        # self.get_logger().info("Initializing Contextual BayesOpt Server")
        super().__init__(
            # node_name = "ContextualBayesOptServer_Node",
            # config_file=config_file,
            server_name=server_name,
            log_file=log_file,
            # anonymous=anonymous,
            # log_level=log_level,
            silent=silent,
            # node_rate=node_rate,
        )
        ## this happens in the parent class
        # self.declare_parameter('contextual_bayesopt_config', ' ')
        # config_file = self.get_parameter('contextual_bayesopt_config').get_parameter_value().string_value

        self.get_logger().debug("[ContextualBayesOptServer] Initialization done")

    @util.count_requests
    def next_parameter_callback(self, goal_handle) -> None:
        goal = goal_handle.request
        self._print_goal(goal) if not self.silent else None
        if self._check_final_iter(goal):
            return  # Do not continue once we reached maximum iterations

        # Obtain the new parameter values.
        result = ContextualBayesOpt.Result()
        # self.get_logger().info(f"dbg - next_param_cb: gaol = {goal}")
        x_new_t = self.bo.next(goal)
        # self.get_logger().info(f"dbg - next_param_cb: x_new_t.tolist() = {x_new_t.tolist()}")
        result.x_new = x_new_t.tolist()
        # self.parameter_server.set_succeeded(result)
        goal_handle.succeed() #TODO: check if this should .accept instead?!
        # self.get_logger().info(f"dbg - next_param_cb: after .succeed")
        self._print_result(result) if not self.silent else None

        # if self.request_count == 15:
        #     tsec1 = timeit.timeit(lambda: self.bo._update_model(goal),number=1)
        #     tsec2 = timeit.timeit(lambda: self.bo._get_next_x(),number=1)
        #     tsec3 = timeit.timeit(lambda: self.bo._log_results(),number=1)
        #     self.get_logger().info(f"--------TIME of _update_model(goal)={tsec1}")
        #     self.get_logger().info(f"--------TIME of _get_next_x()={tsec2}")
        #     self.get_logger().info(f"--------TIME of _log_results()={tsec3}")
        #     tsec5 = timeit.timeit(lambda: self.bo._initialize_model(self.bo.data_handler),number=1)
        #     self.get_logger().info(f"--------TIME of bo._initialize_model(data_handler)={tsec5}")
        #     tsec4 = timeit.timeit(lambda: self.bo._fit_model(),number=1)
        #     self.get_logger().info(f"--------TIME of _fit_model()={tsec4}")
        #     xc = torch.cat((self.bo.x_new, self.bo.context))
        #     tsec6 = timeit.timeit(lambda: self.bo.data_handler.add_xy(x=xc, y=goal.y_new),number=1)
        #     self.get_logger().info(f"--------TIME of data_h.addxy(x,y)={tsec6}")
        #     def test_torch(bo):
        #         torch.cat((bo.x_new, bo.context))
        #     tsec7 = timeit.timeit(lambda: test_torch(self.bo),number=1)
        #     self.get_logger().info(f"--------TIME of data_h.addxy(x,y)={tsec7}")


        return result

    def state_callback(self, goal_handle):

        state = ContextualBayesOptState.Result()
        # Best observed variables
        x_best, c_best, y_best = self.bo.get_best_observation()
        state.x_best = x_best.tolist()
        state.c_best = c_best.tolist()
        state.y_best = float(y_best.item())

        # Posterior mean optimum for a given context
        x_opt, f_opt = self.bo.get_optimal_parameters(goal_handle.request.context)
        state.x_opt = x_opt.tolist()
        state.f_opt = f_opt.item()
        self.get_logger().info(f"[Result] x_best observed: {x_best.tolist()}")
        self.get_logger().info(f"[Result] c_best observed: {c_best.tolist()}")
        self.get_logger().info(f"[Result] y_best observed: {y_best}") 
        self.get_logger().info(f"[Result] optimal estimated params at context={goal_handle.request.context} are:")
        self.get_logger().info(f"[Result] x_opt estimated: {x_opt.tolist()}")
        self.get_logger().info(f"[Result] f_opt estimated: {f_opt}")
        goal_handle.succeed() #old: # self.state_server.set_succeeded(state)

        return state
        

    def _initialize_bayesopt(self, config_file):
        try:
            self.bo = ContextualBayesianOptimization.from_file(config_file,logger = self.get_logger())
        except Exception as e:
            self.get_logger().error(
                f"[ContextualBayesOpt] Something went wrong with initialization: '{e}'"
            )
            # rospy.signal_shutdown("Initialization of ContextualBayesOpt failed.")

    def _initialize_parameter_server(self, server_name):
        """This server obtains new function values and provides new parameters."""
        self.parameter_server = ActionServer(
            self,
            ContextualBayesOpt,
            server_name,
            execute_callback=self.next_parameter_callback,
            # auto_start=False,
        )

    def _initialize_state_server(self, server_name):
        """This server provides the current state/results of BO."""
        self.state_server = ActionServer(
            self,
            ContextualBayesOptState,
            server_name,
            execute_callback=self.state_callback,
            # auto_start=False,
        )

    def _print_goal(self, goal):
        if not self.request_count == 1:
            s = self._log_prefix + f"y_n: {goal.y_new:.3f}"
            s += f", c_(n+1) = {util.iter_to_string(goal.c_new, '.3f')}"
            self.get_logger().info(s)
        else:
            self.get_logger().info(self._log_prefix + f"Discard value: {goal.y_new:.3f}")


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

    contextual_bayesopt_action_server = ContextualBayesOptServer()
  
    try:
        rclpy.spin(contextual_bayesopt_action_server)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()