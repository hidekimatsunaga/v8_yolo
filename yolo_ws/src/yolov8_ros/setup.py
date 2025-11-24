from setuptools import setup

package_name = 'yolov8_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matsunaga-h',
    maintainer_email='your_email@example.com',
    description='YOLOv8 ROS2 wrapper',
    license='MIT',
    extras_require={  # ← ここを間違えるとビルド失敗
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'yolo_node = yolov8_ros.yolo_node:main',
            'yolo_3d_node = yolov8_ros.yolo_3d_node:main',
            'object_detection_node = yolov8_ros.object_detection_node:main',
        ],
    },
)
