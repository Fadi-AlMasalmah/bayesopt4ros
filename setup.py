import os
from setuptools import find_packages, setup
from glob import glob
# from setuptools import setup

package_name = 'bayesopt4ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], #packages=find_packages(exclude=['test']), #
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (f'share/{package_name}/config'),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))), # added for launch files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))), # added for launch files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fadi',
    maintainer_email='fadi.almasalmah94@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bayesopt_action_server = bayesopt4ros.bayesopt_server:main',
            'bayesopt_action_client = bayesopt4ros.bayesopt_client:main',
        ],
    },
)
