
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
import argparse

import torch
from FunctionEncoder import BaseDataset
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from tqdm import trange

LEG_LENGTH = (0.2 **2 + 0.2**2)**0.5
ANKLE_LENGTH = (0.4 **2 + 0.4**2)**0.5
class VariableAntEnv(gym.Env):


    @staticmethod
    def default_dynamics_variable_ranges():
        leg_lengths = (0.2 **2 + 0.2**2)**0.5
        ankle_lengths = (0.4 **2 + 0.4**2)**0.5
        gear = 60
        return {
            "front_left_leg_length": (leg_lengths, leg_lengths),
            "front_left_foot_length":(ankle_lengths, ankle_lengths),
            "front_right_leg_length":(leg_lengths, leg_lengths),
            "front_right_foot_length":(ankle_lengths, ankle_lengths),
            "back_left_leg_length":(leg_lengths, leg_lengths),
            "back_left_foot_length":(ankle_lengths, ankle_lengths),
            "back_right_leg_length":(leg_lengths, leg_lengths),
            "back_right_foot_length":(ankle_lengths, ankle_lengths),
            "front_left_gear":(gear, gear),
            "front_right_gear":(gear, gear),
            "back_left_gear":(gear, gear),
            "back_right_gear":(gear, gear),
            "front_left_ankle_gear":(gear, gear),
            "front_right_ankle_gear":(gear, gear),
            "back_left_ankle_gear":(gear, gear),
            "back_right_ankle_gear":(gear, gear),
        }

    def __init__(self, dynamics_variable_ranges:Dict, *env_args, **env_kwargs):
        '''

        :param dynamics_variable_ranges: A dictionary of ranges for the dynamics variables. The following
        keys are optional. The default values are used if not specified. Provide as a tuple of (lower, upper).
        '''
        super().__init__()
        self.env_args = env_args
        self.env_kwargs = env_kwargs
        self.env_kwargs["terminate_when_unhealthy"] = False
        self.env_kwargs["exclude_current_positions_from_observation"] = False
        self.env_kwargs["include_cfrc_ext_in_observation"] = False

        self.render_mode = env_kwargs.get('render_mode', None)
        self.dynamics_variable_ranges = dynamics_variable_ranges

        # append defaults if not specified
        defaults = self.default_dynamics_variable_ranges()
        for k, v in defaults.items():
            if k not in self.dynamics_variable_ranges:
                self.dynamics_variable_ranges[k] = v

        # placeholder variable
        self.env =  gym.make('Ant-v5', *self.env_args, **self.env_kwargs)
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space[:29]
        self.observation_space = Box(low=self.env.observation_space.low[:29], high=self.env.observation_space.high[:29], dtype=self.env.observation_space.dtype)

        # path to write xml to
        current_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.xml_path = '/tmp/ants'
        self.xml_name =  f'{current_date_time}.xml'
        os.makedirs(self.xml_path, exist_ok=True)

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        reset_hps:bool = True,
    ):
        # sample env parameters
        if reset_hps:
            front_left_leg_length = np.random.uniform(*self.dynamics_variable_ranges['front_left_leg_length'])
            front_left_foot_length = np.random.uniform(*self.dynamics_variable_ranges['front_left_foot_length'])
            front_right_leg_length = np.random.uniform(*self.dynamics_variable_ranges['front_right_leg_length'])
            front_right_foot_length = np.random.uniform(*self.dynamics_variable_ranges['front_right_foot_length'])
            back_left_leg_length = np.random.uniform(*self.dynamics_variable_ranges['back_left_leg_length'])
            back_left_foot_length = np.random.uniform(*self.dynamics_variable_ranges['back_left_foot_length'])
            back_right_leg_length = np.random.uniform(*self.dynamics_variable_ranges['back_right_leg_length'])
            back_right_foot_length = np.random.uniform(*self.dynamics_variable_ranges['back_right_foot_length'])

            front_left_gear = np.random.uniform(*self.dynamics_variable_ranges['front_left_gear'])
            front_right_gear = np.random.uniform(*self.dynamics_variable_ranges['front_right_gear'])
            back_left_gear = np.random.uniform(*self.dynamics_variable_ranges['back_left_gear'])
            back_right_gear = np.random.uniform(*self.dynamics_variable_ranges['back_right_gear'])
            front_left_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['front_left_ankle_gear'])
            front_right_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['front_right_ankle_gear'])
            back_left_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['back_left_ankle_gear'])
            back_right_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['back_right_ankle_gear'])

            # load env
            current_dynamics_dict = {
                'front_left_leg_length': front_left_leg_length,
                'front_left_foot_length': front_left_foot_length,
                'front_right_leg_length': front_right_leg_length,
                'front_right_foot_length': front_right_foot_length,
                'back_left_leg_length': back_left_leg_length,
                'back_left_foot_length': back_left_foot_length,
                'back_right_leg_length': back_right_leg_length,
                'back_right_foot_length': back_right_foot_length,
                'front_left_gear': front_left_gear,
                'front_right_gear': front_right_gear,
                'back_left_gear': back_left_gear,
                'back_right_gear': back_right_gear,
                'front_left_ankle_gear': front_left_ankle_gear,
                'front_right_ankle_gear': front_right_ankle_gear,
                'back_left_ankle_gear': back_left_ankle_gear,
                'back_right_ankle_gear': back_right_ankle_gear,
            }

            # create xml file for these parameters
            path = self.create_xml_file(front_left_leg_length, front_left_foot_length, front_right_leg_length, front_right_foot_length, back_left_leg_length, back_left_foot_length, back_right_leg_length, back_right_foot_length, front_left_gear, front_right_gear, back_left_gear, back_right_gear, front_left_ankle_gear, front_right_ankle_gear, back_left_ankle_gear, back_right_ankle_gear,)
            self.current_dynamics_dict = current_dynamics_dict
            self.env = gym.make('Ant-v5', xml_file=path, *self.env_args, **self.env_kwargs)

        # return observation
        state, info = self.env.reset()
        info["dynamics"] = self.current_dynamics_dict
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        info["dynamics"] = self.current_dynamics_dict
        return next_state, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def create_xml_file(self,
                        front_left_leg_length, front_left_foot_length,
                        front_right_leg_length, front_right_foot_length,
                        back_left_leg_length, back_left_foot_length,
                        back_right_leg_length, back_right_foot_length,
                        front_left_gear, front_right_gear,
                        back_left_gear, back_right_gear,
                        front_left_ankle_gear, front_right_ankle_gear,
                        back_left_ankle_gear, back_right_ankle_gear,
                        ):
        file_string = f"""<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="{0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="{0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {0.4 * front_left_foot_length / ANKLE_LENGTH} {0.4 * front_left_foot_length / ANKLE_LENGTH} 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{0.4 * front_right_foot_length / ANKLE_LENGTH} {0.4 * front_right_foot_length / ANKLE_LENGTH} 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{0.4 * back_left_foot_length / ANKLE_LENGTH} -{0.4 * back_left_foot_length / ANKLE_LENGTH} 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="{0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="{0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {0.4 * back_right_foot_length / ANKLE_LENGTH} -{0.4 * back_right_foot_length / ANKLE_LENGTH} 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="{back_right_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="{back_right_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="{front_left_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="{front_left_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="{front_right_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="{front_right_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="{back_left_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="{back_left_ankle_gear}"/>
  </actuator>
</mujoco>
"""
        with open(os.path.join(self.xml_path, self.xml_name), 'w') as f:
            f.write(file_string)
        return os.path.join(self.xml_path, self.xml_name)


    def set_state(self, qpos, qvel):
        self.env.env.env.env.set_state(qpos, qvel)

    def get_state(self):
        ant_env = self.env.env.env.env
        qpos, qvel = ant_env.data.qpos, ant_env.data.qvel
        obs = ant_env._get_obs()
        return obs[:29], (qpos, qvel)




# generate train and type1 parameter ranges
train_type1_parameters = {}
default_sizes = VariableAntEnv.default_dynamics_variable_ranges()
for k, v in default_sizes.items():
    train_type1_parameters[k] = (v[0]*0.5, v[1]*1.5)
    # if "length" in k:
    #     train_type1_parameters[k] = (v[0]*0.5, v[1]*1.5)
    # else:
    #     train_type1_parameters[k] = (v[0]*1.0, v[1]*1.0)
# type 2 will be linear combinations of these parameter dynamics (not the parameters themselves)

# generate type3 parameter ranges
train_type3_parameters = {}
for k, v in default_sizes.items():
    train_type1_parameters[k] = (v[0]*2.0, v[1]*2.5)
    # if "length" in k:
    #     train_type1_parameters[k] = (v[0]*2.0, v[1]*2.5)
    # else:
    #     train_type1_parameters[k] = (v[0]*1.0, v[1]*1.0)








def visualize():
    hps = train_type1_parameters
    env = VariableAntEnv(hps, render_mode='human')
    env2 = VariableAntEnv(hps, render_mode='human')
    env2.reset()
    # loop and render to make sure its working
    try:
        for _ in range(10_000):
            if _ % 100 == 0:
                env.close()
                env2.close()
                obs, info = env.reset()

            # copy state over
            env2.set_state(*env.get_state()[1])
            obs2 = env2.get_state()[0]

            # render
            env.render()
            time.sleep(0.01)

            # step envs
            n_obs, r, term, trunc, info = env.step(env.action_space.sample())
            n_obs2, r, term, trunc, info = env2.step(env.action_space.sample())

            # see if we can mimic the state
            env2.render()
    except KeyboardInterrupt:
        pass
    env.close()
    env2.close()

def collect_type1_data(num_functions, params, episode_length=1000, n_examples=200) -> dict:
    # create env
    assert params in ["type1", "type3"]
    if params == "type1":
        hps = train_type1_parameters
    else:
        hps = train_type3_parameters
    env = VariableAntEnv(hps, render_mode="rgb_array")

    # create a video of the environment

    # init data buffers
    states = np.zeros((num_functions, episode_length, env.observation_space.shape[0]))
    actions = np.zeros((num_functions, episode_length, env.action_space.shape[0]))
    next_states = np.zeros((num_functions, episode_length, env.observation_space.shape[0]))
    hidden_params = np.zeros((num_functions, len(hps)))

    # collect data
    for episode in trange(num_functions):
        state, info = env.reset()
        hidden_params[episode] = np.array([info["dynamics"][k] for k in hps.keys()])
        for step in range(episode_length):
            # step env
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            # save data
            states[episode, step] = state
            actions[episode, step] = action
            next_states[episode, step] = next_state

            # continue
            state = next_state
    env.close()

    # create dictionary of example xs, example ys, xs, ys.
    # the goal is to predict the next state from the current state.
    all_xs = np.concatenate([states, actions], axis=-1)
    all_ys = next_states

    # take first section to be examples
    example_xs = all_xs[:, :n_examples]
    example_ys = all_ys[:, :n_examples]
    xs = all_xs[:, n_examples:]
    ys = all_ys[:, n_examples:]

    # create torch dict and return
    results = {"example_xs": example_xs, "example_ys": example_ys, "xs": xs, "ys": ys, "hidden_params": hidden_params}
    results = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    return results


def collect_type2_data(num_functions, episode_length=1000, n_examples=200) -> dict:
    # default params
    hps = train_type1_parameters

    # we will be taking a synthetic linear combination of these two environments.
    env = VariableAntEnv(hps)
    env2 = VariableAntEnv(hps)

    # init data buffers
    states = np.zeros((num_functions, episode_length, env.observation_space.shape[0]))
    actions = np.zeros((num_functions, episode_length, env.action_space.shape[0]))
    next_states = np.zeros((num_functions, episode_length, env.observation_space.shape[0]))
    hidden_params = np.zeros((num_functions, 2, len(hps)))
    weights = np.zeros((num_functions, 2))

    # collect data
    for episode in trange(num_functions):
        # sample values between -1 and 1 to take a linear combination of the two environments.
        alpha = np.random.uniform(-1.0, 1.0)
        beta = np.random.uniform(-1.0, 1.0)

        # reset both envs
        state, info = env.reset()
        state2, info2 = env2.reset()
        env2.set_state(*env.get_state()[1])
        state2 = env2.get_state()[0]
        assert np.allclose(state, state2)

        # store hidden params
        hidden_params[episode, 0] = np.array([info["dynamics"][k] for k in hps.keys()])
        hidden_params[episode, 1] = np.array([info2["dynamics"][k] for k in hps.keys()])
        weights[episode] = np.array([alpha, beta])

        # run an episode
        for step in range(episode_length):
            # step env
            action = env.action_space.sample()
            assert (env.get_state()[0] - env2.get_state()[0] < 1e-9).all()
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state2, reward, terminated, truncated, info = env2.step(action)
            assert (next_state != next_state2).any()

            # save data
            states[episode, step] = state
            actions[episode, step] = action
            next_states[episode, step] = alpha * next_state + beta * next_state2

            # continue
            state = next_state
            env2.set_state(*env.get_state()[1])
    env.close()
    env2.close()

    # create dictionary of example xs, example ys, xs, ys.
    # the goal is to predict the next state from the current state.
    all_xs = np.concatenate([states, actions], axis=-1)
    all_ys = next_states

    # take first section to be examples
    example_xs = all_xs[:, :n_examples]
    example_ys = all_ys[:, :n_examples]
    xs = all_xs[:, n_examples:]
    ys = all_ys[:, n_examples:]

    # create torch dict and return
    results = {"example_xs": example_xs, "example_ys": example_ys, "xs": xs, "ys": ys, "hidden_params": hidden_params, "weights": weights}
    results = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    return results


def collect_data(force=False):
    # get save dir
    this_file_path = os.path.abspath(__file__)
    save_dir = os.path.join(os.path.dirname(this_file_path), 'MuJoCo')
    os.makedirs(save_dir, exist_ok=True)

    # if data already exists, exit
    if not force and (os.path.exists(os.path.join(save_dir, 'train.pt')) and
        os.path.exists(os.path.join(save_dir, 'type1.pt')) and
        os.path.exists(os.path.join(save_dir, 'type2.pt')) and
        os.path.exists(os.path.join(save_dir, 'type3.pt'))):
        print("MuJoCo data already exists. Skipping.")
        return

    # gather data
    train = collect_type1_data(num_functions=1000, params="type1")
    type1 = collect_type1_data(num_functions=200, params="type1") # same distribution as training, but different parameters
    type2 = collect_type2_data(num_functions=200) # synthetic data for linear transfer. Not necessarily a feasible trajectory.
    type3 = collect_type1_data(num_functions=200, params="type3") # OOD paramaters.

    # normalize
    xs_means = train['xs'].mean(dim=(0, 1), keepdim=True)
    xs_stds = train['xs'].std(dim=(0, 1), keepdim=True)
    ys_means = xs_means[:, :, :29]
    ys_stds = xs_stds[:, :, :29] # note the first 29 dimensions are the state space, which is the same as the ys. We reuse the normalization so that we can compare the state-next state later in the same units.
    hp_means = train['hidden_params'].mean(dim=(0), keepdim=True) # this is for the oracle. No other algs use this.
    hp_stds = train['hidden_params'].std(dim=(0), keepdim=True)
    for dataset in [train, type1, type2, type3]:
        dataset['xs'] = (dataset['xs'] - xs_means) / xs_stds
        dataset['ys'] = (dataset['ys'] - ys_means) / ys_stds
        dataset['example_xs'] = (dataset['example_xs'] - xs_means) / xs_stds
        dataset['example_ys'] = (dataset['example_ys'] - ys_means) / ys_stds
        dataset['hidden_params'] = (dataset['hidden_params'] - hp_means) / hp_stds

    # save it
    torch.save(train, os.path.join(save_dir, 'train.pt'))
    torch.save(type1, os.path.join(save_dir, 'type1.pt'))
    torch.save(type2, os.path.join(save_dir, 'type2.pt'))
    torch.save(type3, os.path.join(save_dir, 'type3.pt'))




class MujoCoAntDataset(BaseDataset):

    def __init__(self, dataset_type, device, n_examples, n_functions=10):
        # load data
        path = os.path.join(os.path.dirname(__file__), 'MuJoCo')
        if not (os.path.exists(os.path.join(path, 'train.pt')) and
                os.path.exists(os.path.join(path, 'type1.pt')) and
                os.path.exists(os.path.join(path, 'type2.pt')) and
                os.path.exists(os.path.join(path, 'type3.pt'))):
            raise FileNotFoundError("MuJoCo data not found. Please run 'src/datasets/MuJoCoAnt.py' to collect data.")
        if dataset_type == 'train':
            data = torch.load(os.path.join(path, 'train.pt'), weights_only=True)
        elif dataset_type == 'type1':
            data = torch.load(os.path.join(path, 'type1.pt'), weights_only=True)
        elif dataset_type == 'type2':
            data = torch.load(os.path.join(path, 'type2.pt'), weights_only=True)
        elif dataset_type == 'type3':
            data = torch.load(os.path.join(path, 'type3.pt'), weights_only=True)
        else:
            raise ValueError(f"Dataset type {dataset_type} not recognized.")
        self.dataset_type = dataset_type

        # fetch data
        self.example_xs = data['example_xs'].to(device)
        self.example_ys = data['example_ys'].to(device)
        self.xs = data['xs'].to(device)
        self.ys = data['ys'].to(device)
        self.hidden_params = data['hidden_params'].to(device)
        if dataset_type == 'type2':
            self.weights = data['weights'].to(device)

        # init base class
        n_inputs = self.example_xs.shape[-1]
        n_outputs = self.example_ys.shape[-1]
        super().__init__(input_size=(n_inputs,),
                         output_size=(n_outputs,),
                         data_type="deterministic",
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=self.xs.shape[1],
                         device=device)

        # oracle info
        self.oracle_size = self.hidden_params.shape[-1]


    def sample(self) -> Tuple[  torch.tensor,
                            torch.tensor,
                            torch.tensor,
                            torch.tensor,
                            dict]:
        # sample random functions
        function_indicies = torch.randint(0, self.n_functions, (self.n_functions,), device=self.device)
        examples_xs = self.example_xs[function_indicies]
        examples_ys = self.example_ys[function_indicies]
        xs = self.xs[function_indicies]
        ys = self.ys[function_indicies]

        # get oracle info
        if self.dataset_type == 'type2':
            hp1 = self.hidden_params[function_indicies, 0]
            hp2 = self.hidden_params[function_indicies, 1]
            weights = self.weights[function_indicies]
            combined_hps = weights[:, 0].unsqueeze(-1) * hp1 + weights[:, 1].unsqueeze(-1) * hp2
            oracle_info = combined_hps
        else:
            oracle_info = self.hidden_params[function_indicies]

        # maybe downsample number of example points
        if self.n_examples < self.example_xs.shape[1]:
            example_indicies = torch.randint(0, self.example_xs.shape[1], (self.n_examples,), device=self.device)
            examples_xs = examples_xs[:, example_indicies]
            examples_ys = examples_ys[:, example_indicies]

        info = {"oracle_inputs": oracle_info}
        return examples_xs, examples_ys, xs, ys, info


def get_ant_datasets(device, n_examples, n_functions):
    train_dataset = MujoCoAntDataset('train', device, n_examples, n_functions=n_functions)
    type1_dataset = MujoCoAntDataset('type1', device, n_examples, n_functions=n_functions)
    type2_dataset = MujoCoAntDataset('type2', device, n_examples, n_functions=n_functions)
    type3_dataset = MujoCoAntDataset('type3', device, n_examples, n_functions=n_functions)
    return train_dataset, type1_dataset, type2_dataset, type3_dataset

def plot_ant(xs, ys, y_hats, example_xs, example_ys, save_dir, type_i, info):
    labels = [
        "z-cord torso",
        "w-orient torso",
        "x-orient torso",
        "y-orient torso",
        "z-orient torso",
        "torso-hip front left",
        "front left ankle",
        "torso-hip front right",
        "front right ankle",
        "torso-hip back left",
        "back left ankle",
        "torso-hip back right",
        "back right ankle",
        "x-vel torso",
        "y-vel torso",
        "z-vel torso",
        "x-ang vel torso",
        "y-ang vel torso",
        "z-ang vel torso",
        "ang vel torso-hip front left",
        "ang vel front left ankle",
        "ang vel torso-hip front right",
        "ang vel front right ankle",
        "ang vel torso-hip back left",
        "ang vel back left ankle",
        "ang vel torso-hip back right",
        "ang vel back right ankle",
        "x-cord torso (excluded)",
        "y-cord torso (excluded)"
    ]
    for traj in range(ys.shape[0]):
        # plot each output dimension
        fig, axs = plt.subplots(5, 6, figsize=(20, 20))
        axs = axs.flatten()
        for i in range(ys.shape[-1]):
            axs[i].plot(ys[traj, :, i], label='true')
            axs[i].plot(y_hats[traj, :, i], label='pred')
            axs[i].set_title(labels[i])
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"ant_{type_i}_{traj}.png"))
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    vis = args.visualize

    # render the env
    if vis:
        visualize()
    # collect and save data
    else:
        collect_data(args.force)
