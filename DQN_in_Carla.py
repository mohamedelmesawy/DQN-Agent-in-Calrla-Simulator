# import ram_utils.DeepLapV3 as DeepLapV3
# from ram_utils.DeepLapV3 import predict_segmentation
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.backend import backend

import tensorflow as tf

from threading import Thread
from tqdm import tqdm

# from subprocess import call
# import subprocess
# call(["python", "generate_traffic.py"])
# p = subprocess.Popen([sys.executable, 'generate_traffic.py'],
#                                     stdout=subprocess.PIPE,
#                                     stderr=subprocess.STDOUT)

# import os
# os.system('python generate_traffic.py')
####################################################
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
######################################################

from carla import ColorConverter as cc
import carla

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 15  #RAM
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

ACTION_SPACE = 4

# MEMORY_FRACTION = 1
MEMORY_FRACTION = 0.7
MIN_REWARD = -100

EPISODES = 100_000  # Check me
# EPISODES = 30

DISCOUNT = 0.99

# We use a decaying epsilon-greedy policy where the value of epsilon will be decaying over
# time as we don't want to explore forever. So, over time, our policy will be exploiting only good actions:
epsilon = 1
EPSILON_DECAY = 0.99975  # 0.9975 99975  # ram
MIN_EPSILON = 0.05

AGGREGATE_STATS_EVERY = 10  # Check me

# DEEP_LAP_MODEL = DeepLapV3.DeepLabModel(
#     './ram_utils/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(
            self.log_dir)  # tf.summary.FileWriter(self.log_dir)
    # Overriding this method to stop creating default log writer

    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items().sort(reversed=True):
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 0.75 # RAm
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    front_camera_all_windows = np.zeros([im_height, im_width * 2, 3])

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(100.0)
        self.world = self.client.get_world()
        # print(self.client.get_available_maps())
        # self.world = self.client.load_world('Town10HD')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        try:
            self.collision_hist = []
            self.actor_list = []

            self.transform = random.choice(
                self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)

            transform = carla.Transform(carla.Location(x=2.5, z=0.7))

            # 01 + 02 RGB CAMERA + Predictd Sigmentation ###############################################
            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')

            self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
            self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
            self.rgb_cam.set_attribute("fov", f"110")

            self.sensor = self.world.spawn_actor(
                self.rgb_cam, transform, attach_to=self.vehicle)
            self.actor_list.append(self.sensor)
            self.sensor.listen(lambda image: self.process_img(image))

            # 03 SEMANTIC SEGMENTATION CAMERA ################################
            self.semantic_segmentation_cam = self.blueprint_library.find(
                'sensor.camera.semantic_segmentation')
            self.semantic_segmentation_cam.set_attribute(
                "image_size_x", f"{self.im_width}")
            self.semantic_segmentation_cam.set_attribute(
                "image_size_y", f"{self.im_height}")
            self.semantic_segmentation_cam.set_attribute("fov", f"110")

            self.sensor_seg = self.world.spawn_actor(
                self.semantic_segmentation_cam, transform, attach_to=self.vehicle)

            self.actor_list.append(self.sensor_seg)
            self.sensor_seg.listen(
                lambda image: self.process_segmented_img(image))

            # VEHICLE ######################################################
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.0))

            time.sleep(4)

            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(
                colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))

            # while self.front_camera is None:
            while self.front_camera_seg is None:  # modified by RAM
                time.sleep(0.01)

            self.episode_start = time.time()
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.0))

            return self.front_camera_seg  # modified by RAM
            # return self.front_camera
        except:
            return self.reset()

    def collision_data(self, event):
        self.collision_hist.append(event)

    ##### 01 + 02  RGB CAMERA - Predicted Segmented Image ####################
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        # Setting the Left Image
        self.front_camera_rgb = i3
        self.front_camera_all_windows[:, :self.im_width] = i3/255.

    # ##### 03 SEGMENTATION CAMERA - Ground Truth ####################
    def process_segmented_img(self, seg_image):
        seg_image.convert(cc.CityScapesPalette)
        array = np.frombuffer(seg_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (seg_image.height, seg_image.width, 4))
        array = array[:, :, :3]
        # array = array[:, :, ::-1]

        # Setting the Right Image
        self.front_camera_seg = array
        self.front_camera_all_windows[:, self.im_width:] = array/255.

        if self.SHOW_CAM:
            cv2.imshow("FRONT CAMERA ALL WINDOWS",
                       self.front_camera_all_windows)
            cv2.waitKey(1)

    ##### STEP FUNCTION ####################
    def step(self, action):
        ''' # Ram
        - ALL actions right,left,forward
        - handle for the observation, possible collision, and reward
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = - 1 * self.STEER_AMT)) # left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer = 0.0)) # Half-throttle
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer= 0)) # Full-throttle
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = 1 * self.STEER_AMT)) # right
            
        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(
        #         throttle=1.0, steer=-1*self.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(
        #         carla.VehicleControl(throttle=1.0, steer=0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(
        #         throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))


        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif kmh < 35:
            done = False
            reward = -15
        elif kmh > 120:
            done = False
            reward = -10
        else:
            done = False
            reward = 15
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
    
        # if len(self.collision_hist) != 0:
        #     done = True
        #     reward = -100
        # elif kmh < 50:
        #     done = False
        #     reward = -1
        # else:
        #     done = False
        #     reward = 1

        # Is the episode over already ?
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        return self.front_camera_seg, reward, done, None
        # return self.front_camera, reward, done, None  # modified by RAM


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs2/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False,
                              input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Ram  --> 4 Actions
        predictions = Dense(ACTION_SPACE, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(
            learning_rate=0.005), metrics=["accuracy"]) #Ram lr=0.001

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # minibatch is list of rows each one is  transition ==> (current_state, action, reward, new_state, done)
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0]
                                   for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(
                current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] 
                                       for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(
                new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            tf.enable_eager_execution()
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE,
                           verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None,
                           )

        ########## MESAWY ##########
        # # Fit on all samples as one batch, log only on terminal state
        # self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE,
        #                verbose=0, shuffle=False)
        print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(
            size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


if __name__ == '__main__':
    FPS = 20
    # For stats
    ep_rewards = [MIN_REWARD]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    # tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=MEMORY_FRACTION)

    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=gpu_options))

    tf.compat.v1.keras.backend.set_session(session)

    print('--------------------------------------------')

    # Create models folder
    if not os.path.isdir('models2'):
        os.makedirs('models2')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    # trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    # trainer_thread.start()

    # while not agent.training_initialized:
    #     time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # try:

        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, ACTION_SPACE)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory(
                (current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            print('Killing ', actor)
            carla.command.DestroyActor(actor)
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(
                ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                print(f"Saving model {episode+1} ....")
                agent.model.save(
                    f'models2/{MODEL_NAME}__{max_reward:_>10.2f}max_{average_reward:_>10.2f}avg_{min_reward:_>7.2f}min_{EPISODES:_>4.0f}episodes__{int(time.time())}.model')
                    # f'models2/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if not episode % 1000:
            print(f"RAM Saving model {episode+1} ....")
            agent.model.save(
                f'models2/{MODEL_NAME}__{max_reward:_>10.2f}max_{average_reward:_>10.2f}avg_{min_reward:_>7.2f}min_{EPISODES:_>4.0f}episodes__{int(time.time())}.model')
                # f'models2/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    # trainer_thread.join()
    agent.model.save(
        f'models2/{MODEL_NAME}__{max_reward:_>10.2f}max_{average_reward:_>10.2f}avg_{min_reward:_>7.2f}min_{EPISODES:_>4.0f}episodes__{int(time.time())}.model')
        # f'models2/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # p.send_signal(15)
    # print('MESSI', p)

################
## Load Model ##
################
# MODEL_PATH = 'models2/Xception__-118.00max_-179.10avg_-250.00min__1566603992.model'

# if __name__ == '__main__':

#     # Memory fraction
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#     backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

#     # Load the model
#     model = load_model(MODEL_PATH)

#     # Create environment
#     env = CarEnv()

#     # For agent speed measurements - keeps last 60 frametimes
#     fps_counter = deque(maxlen=60)

#     # Initialize predictions - first prediction takes longer as of initialization that has to be done
#     # It's better to do a first prediction then before we start iterating over episode steps
#     model.predict(np.ones((1, env.im_height, env.im_width, 3)))

#     # Loop over episodes
#     while True:

#         print('Restarting episode')

#         # Reset environment and get initial state
#         current_state = env.reset()
#         env.collision_hist = []

#         done = False

#         # Loop over steps
#         while True:

#             # For FPS counter
#             step_start = time.time()

#             # Show current frame
#             cv2.imshow(f'Agent - preview', current_state)
#             cv2.waitKey(1)

#             # Predict an action based on current observation space
#             qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
#             action = np.argmax(qs)

#             # Step environment (additional flag informs environment to not break an episode by time limit)
#             new_state, reward, done, _ = env.step(action)

#             # Set current step for next loop iteration
#             current_state = new_state

#             # If done - agent crashed, break an episode
#             if done:
#                 break

#             # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
#             frame_time = time.time() - step_start
#             fps_counter.append(frame_time)
#             print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

#         # Destroy an actor at end of episode
#         for actor in env.actor_list:
#             actor.destroy()
