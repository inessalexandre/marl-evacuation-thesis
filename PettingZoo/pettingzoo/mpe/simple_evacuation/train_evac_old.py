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
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd


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


class EarlyStopOnAllSuccessCallback(BaseCallback):
    def __init__(self, num_agents, verbose=0):
        super().__init__(verbose)
        self.num_agents = num_agents
        self.episode_rewards = []
        self.episode_lengths = []
        self.evacuation_times = []
        self.timesteps = []
        self.current_rewards = []
        self.current_length = 0
        self.evacuation_time_recorded = False
        self.all_agents_evacuation_time = 0

    def _on_rollout_start(self) -> None:
        self.current_rewards = []
        self.current_length = 0
        self.evacuation_time_recorded = False
        self.all_agents_evacuation_time = 0

    def get_log(self):
        """Retorna DataFrame com as métricas coletadas"""
        return pd.DataFrame({
            "timestep": self.timesteps,
            "reward": self.episode_rewards,
            "episode_length": self.episode_lengths,
            "evacuation_time": self.evacuation_times,
            "evacuation_efficiency": [e / l if l > 0 else 0 for e, l in
                                      zip(self.evacuation_times, self.episode_lengths)]
        })

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals.get("infos", [])
        dones = self.locals["dones"]

        if isinstance(rewards, (list, tuple, np.ndarray)):
            reward_sum = np.sum(rewards)
        else:
            reward_sum = rewards

        self.current_rewards.append(reward_sum)
        self.current_length += 1

        # Verificar se todos os agentes evacuaram usando infos
        if not self.evacuation_time_recorded and infos:
            try:
                # Para ambientes vectorizados, infos pode ser uma lista de dicts
                if isinstance(infos, list) and len(infos) > 0:
                    first_info = infos[0]
                elif isinstance(infos, dict):
                    first_info = next(iter(infos.values()))
                else:
                    first_info = {}

                if 'all_agents_evacuated' in first_info and first_info['all_agents_evacuated']:
                    self.all_agents_evacuation_time = self.current_length
                    self.evacuation_time_recorded = True
                    if self.verbose > 0:
                        print(f"All agents evacuated in {self.all_agents_evacuation_time} steps")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error checking evacuation status: {e}")

        # Registrar métricas quando episódio termina
        if any(dones):
            total_reward = np.sum(self.current_rewards)
            evacuation_time = (self.all_agents_evacuation_time if self.evacuation_time_recorded
                               else self.current_length)

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(self.current_length)
            self.evacuation_times.append(evacuation_time)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(
                    f"Episode ended: length={self.current_length}, evacuation_time={evacuation_time}, recorded={self.evacuation_time_recorded}")

        return True


def get_algo_config(algo):
    if algo == "dqn":
        return {
            "learning_rate": 0.0001,
            "buffer_size": 50000,        # Buffer menor para reduzir não-estacionariedade
            "learning_starts": 1000,     # Começar a aprender mais cedo
            "target_update_interval": 500, # Updates mais frequentes da target network
            "exploration_fraction": 0.3,   # Exploração mais longa
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "train_freq": 4,
            "gradient_steps": 1,
            "batch_size": 64,            # Batch maior para estabilidade
            "tau": 0.005,               # Soft update em vez de hard update
        }
    elif algo == "ppo":
        return {"learning_rate": 0.0003, "n_steps": 2048}
    elif algo == "a2c":
        return {"learning_rate": 0.0007, "n_steps": 5}
    return {}


def train_dqn_with_sharing(model, env, total_steps):
    """Treinamento DQN com reset periódico do buffer"""
    step = 0
    reset_interval = 25000  # Resetar a cada 25k steps

    while step < total_steps:
        # Treinar por um período
        steps_to_train = min(reset_interval, total_steps - step)
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        step += steps_to_train

        # Reset periódico do replay buffer para reduzir não-estacionariedade
        if step < total_steps and step % reset_interval == 0:
            print(f"Resetting replay buffer at step {step}")
            # Salvar o modelo atual
            temp_path = "temp_model.zip"
            model.save(temp_path)

            # Recriar o modelo com buffer limpo
            algo_class = type(model)
            policy_class = type(model.policy)
            model = algo_class(
                policy_class,
                env,
                verbose=1,
                device="cpu",
                **get_algo_config("dqn")
            )
            # Carregar os pesos da rede neural
            model = model.load(temp_path, env=env)

            # Limpar arquivo temporário
            import os
            os.remove(temp_path)

    return model


def train(env_fn, algo="ppo", total_steps=700000, scenario_id=1, seed=0, **env_kwargs):
    """Treinar modelo com parâmetros simplificados"""
    # Remover parâmetros que não existem mais no ambiente
    clean_kwargs = {k: v for k, v in env_kwargs.items()
                    if k not in ['num_agents', 'num_obstacles', 'num_exits', 'num_landmarks']}

    # Crie o ambiente base e obtenha num_agents ANTES dos wrappers
    env_base = env_fn(scenario_id=scenario_id, **clean_kwargs)
    agent_names = list(env_base.possible_agents)  # ou env_base.agents após reset
    num_agents = len(agent_names)
    env_base.reset(seed=seed)

    # Aplique os wrappers
    env = ss.pad_observations_v0(env_base)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")

    algo_class, policy_class = ALGOS[algo]
    model = algo_class(policy_class, env, verbose=1)
    #model.learn(total_timesteps=total_steps)

    callback = EarlyStopOnAllSuccessCallback(num_agents=num_agents, verbose=1)
    model.learn(total_timesteps=700000, callback=callback)
    reward_log = callback.get_log()
    reward_log["scenario"] = scenario_id
    reward_log["algorithm"] = algo
    reward_log.to_csv(f"logs/reward_{algo}_scenario{scenario_id}.csv", index=False)

    os.makedirs("models", exist_ok=True)
    filename = f"models/{algo}_scenario{scenario_id}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(filename)
    print(f"Model saved to {filename}")

    env.close()
    return filename

def train_dqn_with_sharing_with_callback(model, env, total_steps, callback):
    """Treinamento DQN com reset periódico do buffer e callback"""
    step = 0
    reset_interval = 25000

    while step < total_steps:
        steps_to_train = min(reset_interval, total_steps - step)
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False, callback=callback)
        step += steps_to_train

        if step < total_steps and step % reset_interval == 0:
            print(f"Resetting replay buffer at step {step}")
            # ... resto do código de reset igual

    return model



def evaluate(env_fn, algo="ppo", model_path=None, episodes=10, scenario_id=1, **env_kwargs):
    """Avaliar modelo com métricas corrigidas - parar contagem quando agente atinge landmark"""
    clean_kwargs = {k: v for k, v in env_kwargs.items()
                    if k not in ['num_agents', 'num_obstacles', 'num_exits', 'num_landmarks']}

    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
    os.environ['SDL_VIDEO_CENTERED'] = '0'
    clean_kwargs["render_mode"] = "human"

    env = env_fn(scenario_id=scenario_id, **clean_kwargs)
    env = ss.pad_observations_v0(env)

    if model_path is None:
        try:
            model_path = max(glob.glob(f"models/{algo}_scenario{scenario_id}*.zip"), key=os.path.getctime)
        except ValueError:
            print("No trained model found.")
            return

    model_class, _ = ALGOS[algo]
    model = model_class.load(model_path)

    log_data = []

    for ep in range(episodes):
        obs, _ = env.reset()
        agent_names = list(env.agents)

        # Inicializar métricas
        total_rewards = {agent: 0.0 for agent in agent_names}
        positions = {agent: [] for agent in agent_names}
        distances = {agent: 0.0 for agent in agent_names}
        reached_landmark = {agent: False for agent in agent_names}
        collisions = {agent: 0 for agent in agent_names}
        stationary_counts = {agent: 0 for agent in agent_names}

        # CORREÇÃO: Adicionar métricas específicas para quando atingem landmark
        final_rewards = {agent: 0.0 for agent in agent_names}
        evacuation_times = {agent: 0 for agent in agent_names}
        final_distances = {agent: 0.0 for agent in agent_names}

        # Calcular distância inicial específica para cada agente
        initial_positions = {}
        direct_distances = {}
        landmark_pos = env.unwrapped.world.landmarks[0].state.p_pos

        for agent in agent_names:
            agent_obj = None
            for a in env.unwrapped.world.agents:
                if a.name == agent:
                    agent_obj = a
                    break

            if agent_obj:
                initial_pos = agent_obj.state.p_pos.copy()
                initial_positions[agent] = initial_pos
                positions[agent].append(initial_pos)
                direct_distances[agent] = np.linalg.norm(initial_pos - landmark_pos)
            else:
                pos = obs[agent][:2].copy()
                initial_positions[agent] = pos
                positions[agent].append(pos)
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

                # CORREÇÃO: Verificar se atingiu landmark ANTES de calcular métricas
                agent_obj = None
                for a in env.unwrapped.world.agents:
                    if a.name == agent:
                        agent_obj = a
                        break

                # Verificar se alcançou o landmark PRIMEIRO
                landmark_reached_this_step = False
                if agent_obj and not reached_landmark[agent]:
                    if hasattr(agent_obj, 'goal_rewarded') and agent_obj.goal_rewarded:
                        reached_landmark[agent] = True
                        landmark_reached_this_step = True
                        # CORREÇÃO: Capturar métricas no momento exato que atinge landmark
                        final_rewards[agent] = total_rewards[agent] + rewards[agent]
                        evacuation_times[agent] = steps
                        final_distances[agent] = distances[agent]
                        print(f"Agent {agent} reached landmark at step {steps} with reward {final_rewards[agent]:.2f}")

                # Só continuar calculando se ainda não atingiu landmark
                if not reached_landmark[agent]:
                    total_rewards[agent] += rewards[agent]

                    if agent_obj:
                        curr_pos = agent_obj.state.p_pos.copy()
                    else:
                        curr_pos = obs[agent][:2].copy()

                    positions[agent].append(curr_pos)

                    # Calcular distância percorrida
                    if len(positions[agent]) > 1:
                        prev_pos = positions[agent][-2]
                        step_dist = np.linalg.norm(curr_pos - prev_pos)
                        distances[agent] += step_dist

                        if step_dist < movement_threshold:
                            stationary_counts[agent] += 1

                    # Verificar colisões
                    if agent_obj:
                        for ob in env.unwrapped.world.obstacles + env.unwrapped.world.walls:
                            if np.linalg.norm(curr_pos - ob.state.p_pos) < (agent_obj.size + ob.size):
                                collisions[agent] += 1

            # Terminar episódio se todos alcançaram o objetivo
            if all(reached_landmark.values()):
                print("All agents have reached the goal. Ending episode early.")
                break

        # CORREÇÃO: Usar métricas finais capturadas no momento da chegada
        for agent in agent_names:
            # Se não atingiu landmark, usar valores atuais
            if not reached_landmark[agent]:
                final_rewards[agent] = total_rewards[agent]
                evacuation_times[agent] = steps  # Tempo máximo se não completou
                final_distances[agent] = distances[agent]

            # Calcular eficiência do caminho
            path_eff = direct_distances[agent] / final_distances[agent] if final_distances[agent] > 0 else 0
            congestion_ratio = stationary_counts[agent] / evacuation_times[agent] if evacuation_times[agent] > 0 else 0

            log_data.append({
                "episode": ep + 1,
                "agent_id": agent,
                "total_reward": final_rewards[agent],  # CORREÇÃO: Usar reward final
                "success": int(reached_landmark[agent]),
                "evacuation_time": evacuation_times[agent],  # CORREÇÃO: Usar tempo final
                "distance_travelled": final_distances[agent],  # CORREÇÃO: Usar distância final
                "direct_distance": direct_distances[agent],
                "initial_position": json.dumps(initial_positions[agent].tolist(), cls=NumpyEncoder),
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

        # Debug: Mostrar métricas finais
        print("Final metrics per agent:")
        for agent in agent_names:
            print(
                f" {agent}: Time={evacuation_times[agent]}, Reward={final_rewards[agent]:.2f}, Success={reached_landmark[agent]}")

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

        # Estatísticas corrigidas
        successful_agents = df[df['success'] == 1]
        if len(successful_agents) > 0:
            print(f"\nSuccessful agents statistics:")
            print(f"Average evacuation time: {successful_agents['evacuation_time'].mean():.1f} steps")
            print(f"Average final reward: {successful_agents['total_reward'].mean():.2f}")
            print(f"Success rate: {len(successful_agents) / len(df) * 100:.1f}%")
        else:
            print("No successful evacuations recorded.")
    else:
        print("No data collected during evaluation.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=ALGOS.keys(), default="ppo")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = {"max_cycles": 100}

    if args.train:
        train(env_fn, algo=args.algo, total_steps=750000, scenario_id=args.scenario, **config)

    if args.eval:
        evaluate(env_fn, algo=args.algo, episodes=4, scenario_id=args.scenario, **config)