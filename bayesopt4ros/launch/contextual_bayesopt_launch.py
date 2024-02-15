import launch
import launch_ros.actions
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    contextual_bayesopt_config_file = PathJoinSubstitution(
        [
            FindPackageShare("bayesopt4ros"),
            "config",
            "contextual_forrester_ucb.yaml",
        ]
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='bayesopt4ros',
            executable='contextual_bayesopt_action_server', # entry point name in setup.py
            name='ContextualBayesOptServer_Node',
            parameters = [{'config_file': contextual_bayesopt_config_file}])
        ])


