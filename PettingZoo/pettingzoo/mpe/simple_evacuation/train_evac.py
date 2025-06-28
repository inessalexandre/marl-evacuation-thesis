import argparse
import os
import time
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from simple_evacuation import parallel_env as env_fn
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.ppo import MlpPolicy as PPOPolicy
from stable_baselines3.a2c import MlpPolicy as A2CPolicy
from stable_baselines3.dqn import MlpPolicy as DQNPolicy
import supersuit as ss


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


ALGOS = {
    "ppo": (PPO, PPOPolicy),
    "a2c": (A2C, A2CPolicy),
    "dqn": (DQN, DQNPolicy),
}


def train(env_fn, algo="ppo", total_steps=100000, scenario_id=1, seed=0, **env_kwargs):
    """Treinar modelo com parâmetros simplificados"""
    # Remover parâmetros que não existem mais no ambiente
    clean_kwargs = {k: v for k, v in env_kwargs.items()
                    if k not in ['num_agents', 'num_obstacles', 'num_exits', 'num_landmarks']}

    env = env_fn(scenario_id=scenario_id, **clean_kwargs)
    env.reset(seed=seed)
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


def evaluate(env_fn, algo="ppo", model_path=None, episodes=10, scenario_id=1, **env_kwargs):
    """Avaliar modelo com sistema de métricas simplificado"""
    # Configurar ambiente para avaliação
    clean_kwargs = {k: v for k, v in env_kwargs.items()
                    if k not in ['num_agents', 'num_obstacles', 'num_exits', 'num_landmarks']}

    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
    os.environ['SDL_VIDEO_CENTERED'] = '0'
    clean_kwargs["render_mode"] = "human"

    env = env_fn(scenario_id=scenario_id, **clean_kwargs)
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

        total_rewards = {agent: 0.0 for agent in agent_names}
        positions = {agent: [] for agent in agent_names}
        distances = {agent: 0.0 for agent in agent_names}
        reached_landmark = {agent: False for agent in agent_names}
        collisions = {agent: 0 for agent in agent_names}
        stationary_counts = {agent: 0 for agent in agent_names}

        direct_distances = {}
        for agent in agent_names:
            pos = obs[agent][:2].copy()
            positions[agent].append(pos)
            # Calcular distância direta ao landmark principal
            landmark_pos = env.unwrapped.world.landmarks[0].state.p_pos
            direct_distances[agent] = np.linalg.norm(pos - landmark_pos)

        terminated = {agent: False for agent in agent_names}
        truncated = {agent: False for agent in agent_names}
        steps = 0
        movement_threshold = 0.01

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
                if terminated[agent] or truncated[agent]:
                    continue

                total_rewards[agent] += rewards[agent]
                curr_pos = obs[agent][:2].copy()
                positions[agent].append(curr_pos)

                if len(positions[agent]) > 1:
                    prev_pos = positions[agent][-2]
                    step_dist = np.linalg.norm(curr_pos - prev_pos)
                    distances[agent] += step_dist

                    if step_dist < movement_threshold:
                        stationary_counts[agent] += 1

                agent_obj = None
                for a in env.unwrapped.world.agents:
                    if a.name == agent:
                        agent_obj = a
                        break

                if agent_obj:
                    for ob in env.unwrapped.world.obstacles:
                        if np.linalg.norm(curr_pos - ob.state.p_pos) < (agent_obj.size + ob.size):
                            collisions[agent] += 1

                    if not reached_landmark[agent] and hasattr(agent_obj, 'goal_rewarded') and agent_obj.goal_rewarded:
                        reached_landmark[agent] = True

            if all(reached_landmark.values()):
                print("All agents have reached the goal. Ending episode early.")
                break


        for agent in agent_names:
            path_eff = direct_distances[agent] / distances[agent] if distances[agent] > 0 else 0
            congestion_ratio = stationary_counts[agent] / steps if steps > 0 else 0

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
                "congestion_ratio": congestion_ratio,
                "scenario_id": scenario_id,
                "algorithm": algo,
                "num_obstacles": len(getattr(env.unwrapped.world, 'obstacles', [])),
                "num_exits": len(getattr(env.unwrapped.world, 'exits', [])),
                "num_agents": len(agent_names),
                "trajectory": json.dumps([p.tolist() for p in positions[agent]], cls=NumpyEncoder),
            })

        print(f"Episode {ep + 1} finished. Success: {sum(reached_landmark.values())}/{len(agent_names)}")

    env.close()

    # Salvar métricas
    if log_data:
        df = pd.DataFrame(log_data)
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = (
            f"logs/{algo}_scenario{scenario_id}"
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
        train(env_fn, algo=args.algo, total_steps=1000, scenario_id=args.scenario, **config)

    if args.eval:
        evaluate(env_fn, algo=args.algo, episodes=4, scenario_id=args.scenario, **config)
