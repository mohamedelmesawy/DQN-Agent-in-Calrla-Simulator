CMD "./CARLA Simulator/CARLA_0.9.13/WindowsNoEditor/CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe" -quality-level Low -fps 20

cd "./CARLA Simulator/CARLA_0.9.13/WindowsNoEditor/PythonAPI/examples"

conda env create -f carla.yml
conda activate carla

python generate_traffic.py -n 70 -w 100

python DQN_in_Carla.py

tensorboard --logdir logs
