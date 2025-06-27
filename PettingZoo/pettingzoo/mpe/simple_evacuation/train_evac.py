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


def train(env_fn, algo="ppo", total_steps=500000, scenario_id=1, seed=0, **env_kwargs):
    env = env_fn(scenario_id=scenario_id, **env_kwargs)
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

        # Métricas existentes
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
        arrival_times = {agent: None for agent in agent_names}

        # Novas métricas
        velocities = {agent: [] for agent in agent_names}
        explored_cells = {agent: set() for agent in agent_names}
        goal_directions = {agent: [] for agent in agent_names}
        proximity_to_others = {agent: [] for agent in agent_names}
        exit_interactions = {agent: [] for agent in agent_names}
        passed_exit = {agent: False for agent in agent_names}
        communication_events = {agent: 0 for agent in agent_names}
        exit_queue_times = {agent: 0 for agent in agent_names}

        # Inicialização
        for agent in agent_names:
            pos = obs[agent][:2].copy()
            positions[agent].append(pos)
            velocities[agent].append(np.array([0.0, 0.0]))

            # Adicionar célula explorada (grid 10x10)
            grid_x = int((pos[0] + 1) * 5)  # Normalizar para grid 10x10
            grid_y = int((pos[1] + 1) * 5)
            explored_cells[agent].add((grid_x, grid_y))

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

                # Calcular velocidade
                if len(positions[agent]) > 1:
                    prev_pos = positions[agent][-2]
                    velocity = curr_pos - prev_pos
                    velocities[agent].append(velocity)

                    step_dist = np.linalg.norm(velocity)
                    distances[agent] += step_dist

                    if step_dist < movement_threshold:
                        stationary_counts[agent] += 1

                # Exploração (grid coverage)
                grid_x = int((curr_pos[0] + 1) * 5)
                grid_y = int((curr_pos[1] + 1) * 5)
                if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                    explored_cells[agent].add((grid_x, grid_y))

                # Direção para o objetivo
                goal_pos = env.unwrapped.world.landmarks[0].state.p_pos
                goal_direction = goal_pos - curr_pos
                if np.linalg.norm(goal_direction) > 0:
                    goal_direction = goal_direction / np.linalg.norm(goal_direction)
                goal_directions[agent].append(goal_direction)

                # Proximidade com outros agentes
                other_distances = []
                for other_agent in agent_names:
                    if other_agent != agent:
                        other_pos = obs[other_agent][:2]
                        dist_to_other = np.linalg.norm(curr_pos - other_pos)
                        other_distances.append(dist_to_other)
                proximity_to_others[agent].append(np.mean(other_distances) if other_distances else 0)

                # Distância ao objetivo
                dist_to_goal = np.linalg.norm(curr_pos - goal_pos)
                distance_to_goal_per_step[agent].append(dist_to_goal)

                # Comunicação
                idx = int(agent.split("_")[1])
                c_msg = env.unwrapped.world.agents[idx].state.c.copy()
                messages[agent].append(c_msg.tolist())

                if np.linalg.norm(c_msg) > 0.01:
                    communication_events[agent] += 1
                    if steps_to_communicate[agent] is None:
                        steps_to_communicate[agent] = steps

                # Verificar colisões com obstáculos
                for ob in env.unwrapped.world.obstacles:
                    if np.linalg.norm(curr_pos - ob.state.p_pos) < (agent_objects[agent].size + ob.size):
                        collisions[agent] += 1

                # Verificar passagem por saídas
                for ex in env.unwrapped.world.exits:
                    if np.linalg.norm(curr_pos - ex.state.p_pos) < (agent_objects[agent].size + ex.size):
                        if not passed_exit[agent]:
                            passed_exit[agent] = True
                            exit_interactions[agent].append(steps)

                # Verificar chegada ao objetivo
                if not reached_landmark[agent] and agent_objects[agent].goal_rewarded:
                    reached_landmark[agent] = True
                    arrival_times[agent] = steps

            if all(reached_landmark.values()):
                print("All agents have reached the goal. Ending episode early.")
                break

        # Calcular métricas finais para cada agente
        for agent in agent_names:
            # Métricas básicas
            path_eff = direct_distances[agent] / distances[agent] if distances[agent] > 0 else 0

            # Coordination score (baseado na sincronização de chegada)
            if arrival_times[agent] is not None:
                all_arrival_times = [t for t in arrival_times.values() if t is not None]
                if len(all_arrival_times) > 1:
                    coordination_score = 1.0 - (np.std(all_arrival_times) / np.mean(all_arrival_times))
                else:
                    coordination_score = 1.0
            else:
                coordination_score = 0.0

            # Exploration coverage (% do grid explorado)
            exploration_coverage = len(explored_cells[agent]) / 100.0  # Grid 10x10 = 100 células

            # Exploration efficiency (cobertura por distância)
            exploration_efficiency = exploration_coverage / distances[agent] if distances[agent] > 0 else 0

            # Congestion ratio (tempo parado / tempo total)
            congestion_ratio = stationary_counts[agent] / steps if steps > 0 else 0

            # Communication effectiveness (mensagens úteis / total)
            useful_messages = sum(1 for msg in messages[agent] if np.linalg.norm(msg) > 0.01)
            communication_effectiveness = useful_messages / len(messages[agent]) if messages[agent] else 0

            # Velocity standard deviation
            if len(velocities[agent]) > 1:
                velocity_magnitudes = [np.linalg.norm(v) for v in velocities[agent]]
                velocity_std = np.std(velocity_magnitudes)
            else:
                velocity_std = 0.0

            # Average goal alignment
            if goal_directions[agent]:
                velocity_directions = []
                for i, vel in enumerate(velocities[agent]):
                    if np.linalg.norm(vel) > 0:
                        vel_dir = vel / np.linalg.norm(vel)
                        if i < len(goal_directions[agent]):
                            alignment = np.dot(vel_dir, goal_directions[agent][i])
                            velocity_directions.append(alignment)
                avg_goal_alignment = np.mean(velocity_directions) if velocity_directions else 0
            else:
                avg_goal_alignment = 0

            # Average proximity to others
            avg_proximity_to_others = np.mean(proximity_to_others[agent]) if proximity_to_others[agent] else 0

            # Exit queue time
            exit_queue_time = exit_interactions[agent][0] if exit_interactions[agent] else None

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
                "num_messages": len(messages[agent]),
                "arrival_time": arrival_times[agent],

                "coordination_score": coordination_score,
                "exploration_coverage": exploration_coverage,
                "exploration_efficiency": exploration_efficiency,
                "congestion_ratio": congestion_ratio,
                "communication_effectiveness": communication_effectiveness,
                "velocity_std": velocity_std,
                "avg_goal_alignment": avg_goal_alignment,
                "avg_proximity_to_others": avg_proximity_to_others,
                "exit_queue_time": exit_queue_time,
                "passed_exit": int(passed_exit[agent]),

                "scenario_id": env.unwrapped.world.scenario_id,
                "algorithm": algo,
                "num_obstacles": len(getattr(env.unwrapped.world, 'obstacles', [])),
                "num_exits": len(getattr(env.unwrapped.world, 'exits', [])),
                "num_agents": len(agent_names),

                "trajectory": json.dumps([p.tolist() for p in positions[agent]], cls=NumpyEncoder),
                "messages": json.dumps(messages[agent], cls=NumpyEncoder),
                "reward_curve": json.dumps(reward_per_step[agent], cls=NumpyEncoder),
                "dist_to_goal_curve": json.dumps(distance_to_goal_per_step[agent], cls=NumpyEncoder),
                "velocity_profile": json.dumps([v.tolist() for v in velocities[agent]], cls=NumpyEncoder),
                "proximity_profile": json.dumps(proximity_to_others[agent], cls=NumpyEncoder),

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
        train(env_fn, algo=args.algo, total_steps=500000, scenario_id=args.scenario, **config)
    if args.eval:
        evaluate(env_fn, algo=args.algo, episodes=4, scenario_id=args.scenario, **config)
