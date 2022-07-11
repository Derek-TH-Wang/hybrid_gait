# hybrid_gait

This repo contain a RL training env for the MIT mini cheetah quadruped robot.

The training task is to train the different gait type according to the various vel input.

The algorithm stucture is hybrid, which means we use RL to train the 4 gait params(gait_type, gait period, gait offset, gait duration). Then the gait params are as the input to the MPC controller, which calculate the joint torque.

We use openai-gym as the RL framework, stable-baselines(PPO) as the RL algorithm lib.

The MPC controller we use the repo of ```https://github.com/Derek-TH-Wang/quadruped_ctrl```

[![Watch the video](https://i.imgur.com/vKb2F1B.png)](https://youtu.be/vt5fpE0bzSY)

## installation
### python
```
sudo apt-get install virtualenv
virtualenv venv --python=/usr/bin/python3.7
source ${path}/bin/active

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
### cpp
```
cd hybrid_gait/quadruped_ctrl/
mkdir build
cd build
cmake ..
make -j
```

## config
see the ```config/training_params.yaml```
```
mode: train or test
model_file: empty or specific file
```

## run

```
python3 hybrid_gait/test.py
```
or use mpiexec:
```
mpiexec -n 4 python3 hybrid_gait/test.py
```




