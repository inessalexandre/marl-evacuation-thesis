import argparse
import os
import time
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from simple_evacuation import parallel_env as env_fn
from stable_baselines3 import PPO, A2C
from stable_baselines3.ppo import MlpPolicy as PPOPolicy
from stable_baselines3.a2c import MlpPolicy as A2CPolicy
from sb3_contrib import QMIX, QMIXPolicy


import supersuit as ss

ALGOS = {
    "ppo": (PPO, PPOPolicy),
    "a2c": (A2C, A2CPolicy),
    "qmix": (QMIX, QMIXPolicy),
}


def train(env_fn, algo="ppo", total_steps=500000, scenario_id=1, seed=0, **env_kwargs):
    env = env_fn(scenario_id=scenario_id, **env_kwargs)
    env.reset(seed=seed)

    if algo == "qmix":
        # Configuração específica para QMIX
        env = ss.pad_observations_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")
        model = QMIX(
            QMIXPolicy,
            env,
            learning_rate=0.0003,
            buffer_size=100000,
            batch_size=32,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            verbose=1
        )
    else:
        # Configuração padrão para outros algoritmos
        env = ss.pad_observations_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")
        algo_class, policy_class = ALGOS[algo]
        model = algo_class(policy_class, env, verbose=1)

    model.learn(total_timesteps=total_steps)

    os.makedirs("models", exist_ok=True)
    filename = f"models/{algo}_scenario{scenario_id}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(filename)
    print(f"Model saved to {filename}")
    env.close()
    return filename


def evaluate(env_fn, algo="ppo", model_path=None, episodes=10, **env_kwargs):
    env_kwargs["render_mode"] = "human"
    env = env_fn(**env_kwargs)
    env = ss.pad_observations_v0(env)

    if model_path is None:
        try:
            model_path = max(glob.glob(f"models/{algo}_*.zip"), key=os.path.getctime)
        except ValueError:
            print("No trained model found.")
            return

    model_class, _ = ALGOS[algo]
    model = model_class.load(model_path)

    log_data = []

    for ep in range(episodes):
        obs, _ = env.reset()
        agent_names = list(env.agents)
        agent_objects = {agent.name: agent for agent in env.unwrapped.world.agents}

        total_rewards = {agent: 0.0 for agent in agent_names}
        positions = {agent: [] for agent in agent_names}
        distances = {agent: 0.0 for agent in agent_names}
        direct_distances = {}
        stationary_counts = {agent: 0 for agent in agent_names}
        movement_threshold = 0.01
        collisions = {agent: 0 for agent in agent_names}
        reached_landmark = {agent: False for agent in agent_names}
        messages = {agent: [] for agent in agent_names}
        reward_per_step = {agent: [] for agent in agent_names}
        distance_to_goal_per_step = {agent: [] for agent in agent_names}
        steps_to_communicate = {agent: None for agent in agent_names}
        arrival_times = {agent: None for agent in agent_names}  # novo

        for agent in agent_names:
            pos = obs[agent][:2].copy()
            positions[agent].append(pos)
            dist = np.linalg.norm(pos - env.unwrapped.world.landmarks[0].state.p_pos)
            direct_distances[agent] = dist
            distance_to_goal_per_step[agent].append(dist)

        terminated = {agent: False for agent in agent_names}
        truncated = {agent: False for agent in agent_names}
        steps = 0

        while not all(terminated.values()) and not all(truncated.values()):
            actions = {
                agent: model.predict(obs[agent], deterministic=True)[0]
                if not terminated[agent] and not truncated[agent]
                else None
                for agent in agent_names
            }

            obs, rewards, terminated, truncated, infos = env.step(actions)
            steps += 1

            for agent in agent_names:
                total_rewards[agent] += rewards[agent]
                reward_per_step[agent].append(rewards[agent])

                curr_pos = obs[agent][:2].copy()
                positions[agent].append(curr_pos)

                if len(positions[agent]) > 1:
                    prev_pos = positions[agent][-2]
                    step_dist = np.linalg.norm(curr_pos - prev_pos)
                    distances[agent] += step_dist
                    if step_dist < movement_threshold:
                        stationary_counts[agent] += 1

                dist_to_goal = np.linalg.norm(curr_pos - env.unwrapped.world.landmarks[0].state.p_pos)
                distance_to_goal_per_step[agent].append(dist_to_goal)

                idx = int(agent.split("_")[1])
                c_msg = env.unwrapped.world.agents[idx].state.c.copy()
                messages[agent].append(c_msg.tolist())

                if steps_to_communicate[agent] is None and np.linalg.norm(c_msg) > 0.01:
                    steps_to_communicate[agent] = steps

                if not reached_landmark[agent] and agent_objects[agent].goal_rewarded:
                    reached_landmark[agent] = True
                    arrival_times[agent] = steps  # novo

            if all(reached_landmark.values()):
                print("All agents have reached the goal. Ending episode early.")
                break

        for agent in agent_names:
            path_eff = direct_distances[agent] / distances[agent] if distances[agent] > 0 else 0
            log_data.append({
                "episode": ep + 1,
                "agent_id": agent,
                "total_reward": total_rewards[agent],
                "success": int(reached_landmark[agent]),
                "evacuation_time": steps,
                "distance_travelled": distances[agent],
                "direct_distance": direct_distances[agent],
                "path_efficiency": path_eff,
                "collisions": collisions[agent],
                "stationary_steps": stationary_counts[agent],
                "steps_to_first_comm": steps_to_communicate[agent],
                "num_messages": len(messages[agent]),  # novo
                "arrival_time": arrival_times[agent],  # novo
                "trajectory": json.dumps([p.tolist() for p in positions[agent]]),
                "messages": json.dumps(messages[agent]),
                "reward_curve": json.dumps(reward_per_step[agent]),
                "dist_to_goal_curve": json.dumps(distance_to_goal_per_step[agent]),
            })

        print(f"Episode {ep + 1} finished. Success: {sum(reached_landmark.values())}/{len(agent_names)}")

    env.close()


    if log_data:
        df = pd.DataFrame(log_data)
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = (
            f"logs/{algo}_scenario{env.unwrapped.world.scenario_id}"
            f"_agents{len(agent_names)}"
            f"_obs{len(getattr(env.unwrapped.world, 'obstacles', []))}"
            f"_exits{len(getattr(env.unwrapped.world, 'exits', []))}"
            f"_{timestamp}.csv"
        )
        df.to_csv(log_filename, index=False)
        print(f"Evaluation metrics saved to {log_filename}")
    else:
        print("No data collected during evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=ALGOS.keys(), default="ppo")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = {"max_cycles": 100}

    if args.train:
        train(env_fn, algo=args.algo, total_steps=500000, scenario_id=args.scenario, **config)
    if args.eval:
        evaluate(env_fn, algo=args.algo, episodes=4, scenario_id=args.scenario, **config)
