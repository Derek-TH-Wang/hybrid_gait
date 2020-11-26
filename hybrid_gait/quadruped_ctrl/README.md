# quadruped_robot

### System requirements:
Ubuntu 18.04, ROS Mellodic  

### Dependency:
use Logitech gamepad to control robot  
```
git clone https://github.com/Derek-TH-Wang/gamepad_ctrl.git
```

### Build
```
cd {your workspace}
catkin make
source devel/setup.bash
```

### Terrain
you can modify the ```config/quadruped_ctrl_cinfig.yaml/terrain``` to deploy different terrains, there are four terrains supported in the simulator now, for example:
```
"plane"
"stairs"
"random1"
"random2"
```

### Robot mode
there are two mode of the robot:
```
rosservice call /robot_mode "cmd: 1"
```

robot mode:
```
0: high performance mode(default)
1: low energy mode(can save approximate 10% energy for now)
```

### Running:
run the gamepad node to control robot:
```
roslaunch gamepad_ctrl gamepad_ctrl.launch
```
run the controller in simulator:  
```
roslaunch quadruped_ctrl quadruped_ctrl.launch
```
you can switch the gait type:  
```
rosservice call /gait_type "cmd: 1"
```

gait type:
```
0:trot
1:bunding
2:pronking
3:random
4:standing
5:trotRunning
6:random2
7:galloping
8:pacing
9:trot (same as 0)
10:walking
11:walking2
```

