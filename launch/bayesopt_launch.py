import launch
import launch_ros.actions
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    bayesopt_config_file = PathJoinSubstitution(
        [
            FindPackageShare("bayesopt4ros"),
            "config",
            "forrester_ucb.yaml",
        ]
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='bayesopt4ros',
            executable='bayesopt_action_server', # entry point name in setup.py
            name='BayesOptServer_Node',
            parameters = [{'bayesopt_config': bayesopt_config_file}])
        ])


