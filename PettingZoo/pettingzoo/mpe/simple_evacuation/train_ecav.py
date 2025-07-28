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


def get_algo_config(algo):
    if algo == "dqn":
        return {
            # === PARÂMETROS CORE ===
            "learning_rate": 0.0005,  # LR ligeiramente maior para comunicação
            "buffer_size": 100000,  # Buffer maior para experiências compartilhadas
            "learning_starts": 1000,  # Início mais tardio para estabilizar comunicação
            "target_update_interval": 500,  # Updates menos frequentes para estabilidade

            # === EXPLORAÇÃO ADAPTADA ===
            "exploration_fraction": 0.5,  # Exploração prolongada para descobrir protocolos
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,  # Exploração mínima mais baixa

            # === TREINAMENTO OTIMIZADO ===
            "train_freq": 4,  # Frequência padrão para comunicação
            "gradient_steps": 1,  # Steps conservadores para estabilidade
            "batch_size": 64,  # Batch menor para comunicação eficaz
            "tau": 0.005,  # Soft updates muito suaves
            "gamma": 0.95,  # Discount ligeiramente menor para comunicação
        }
    elif algo == "a2c":
        return {
            "learning_rate": 0.001,  # Aumentar learning rate
            "n_steps": 10,  # Mais steps para capturar sequências
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,  # Incentivar exploração
            "vf_coef": 0.5,
        }
    return None


def train(env_fn, algo="ppo", total_steps=500000, scenario_id=1, seed=0, **env_kwargs):
    """Treinar modelo com parâmetros simplificados"""
    # Remover parâmetros que não existem mais no ambiente
    clean_kwargs = {k: v for k, v in env_kwargs.items()
                    if k not in ['num_agents', 'num_obstacles', 'num_exits', 'num_landmarks']}

    # Crie o ambiente base e obtenha num_agents ANTES dos wrappers
    env_base = env_fn(scenario_id=scenario_id, **clean_kwargs)
    agent_names = list(env_base.possible_agents)
    num_agents = len(agent_names)
    env_base.reset(seed=seed)

    # Aplique os wrappers
    env = ss.pad_observations_v0(env_base)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class="stable_baselines3")

    # CORREÇÃO: Obter configurações do algoritmo
    algo_class, policy_class = ALGOS[algo]
    algo_config = get_algo_config(algo)

    # Criar modelo com configurações customizadas
    if algo_config:
        model = algo_class(policy_class, env, verbose=1, **algo_config)
    else:
        model = algo_class(policy_class, env, verbose=1)

    model.learn(total_timesteps=total_steps)

    os.makedirs("models", exist_ok=True)
    filename = f"models/{algo}_scenario{scenario_id}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(filename)
    print(f"Model saved to {filename}")

    env.close()
    return filename


def _get_bottleneck_areas(scenario_id):
    """Definir áreas de gargalo por cenário"""
    if scenario_id == 2:
        # Corredor entre x=-0.65 e x=-0.45, y entre -0.2 e 0.2
        return [{"x_min": -0.65, "x_max": -0.45, "y_min": -0.2, "y_max": 0.2}]
    elif scenario_id == 3:
        # Mesmo corredor + áreas ao redor dos obstáculos
        return [
            {"x_min": -0.65, "x_max": -0.45, "y_min": -0.2, "y_max": 0.2},
            {"x_min": 0.25, "x_max": 0.55, "y_min": -0.15, "y_max": 0.15},  # Ao redor do obstáculo central
        ]
    return []


def _is_in_bottleneck(position, bottleneck_areas):
    """Verificar se posição está em área de gargalo"""
    x, y = position
    for area in bottleneck_areas:
        if (area["x_min"] <= x <= area["x_max"] and
                area["y_min"] <= y <= area["y_max"]):
            return True
    return False


def _calculate_coordination_index(positions, agent_names):
    """Calcular índice de coordenação baseado na sincronização de movimentos"""
    if len(agent_names) < 2:
        return 1.0

    # Calcular variância das posições finais como medida de dispersão
    final_positions = []
    for agent in agent_names:
        if len(positions[agent]) > 0:
            final_positions.append(positions[agent][-1])

    if len(final_positions) < 2:
        return 1.0

    final_positions = np.array(final_positions)
    centroid = np.mean(final_positions, axis=0)
    distances_to_centroid = [np.linalg.norm(pos - centroid) for pos in final_positions]
    coordination_variance = np.var(distances_to_centroid)

    # Converter para índice (menor variância = melhor coordenação)
    coordination_index = 1.0 / (1.0 + coordination_variance)
    return coordination_index


def _print_detailed_statistics(df):
    """Imprimir estatísticas detalhadas"""
    successful_agents = df[df['success'] == 1]

    print(f"\n=== DETAILED EVALUATION STATISTICS ===")
    print(f"Total agents evaluated: {len(df)}")
    print(f"Successful evacuations: {len(successful_agents)}")
    print(f"Overall success rate: {len(successful_agents) / len(df) * 100:.1f}%")

    if len(successful_agents) > 0:
        print(f"\n--- SUCCESSFUL AGENTS METRICS ---")
        print(
            f"Average evacuation time: {successful_agents['evacuation_time'].mean():.1f} ± {successful_agents['evacuation_time'].std():.1f} steps")
        print(
            f"Average final reward: {successful_agents['total_reward'].mean():.2f} ± {successful_agents['total_reward'].std():.2f}")
        print(
            f"Average path efficiency: {successful_agents['path_efficiency'].mean():.3f} ± {successful_agents['path_efficiency'].std():.3f}")

        print(f"\n--- COLLISION ANALYSIS ---")
        print(f"Average obstacle collisions: {successful_agents['collisions_obstacle'].mean():.1f}")
        print(f"Average agent collisions: {successful_agents['collisions_agent'].mean():.1f}")
        print(f"Average near-miss events: {successful_agents['near_miss_events'].mean():.1f}")

        print(f"\n--- COORDINATION METRICS ---")
        print(f"Average coordination index: {successful_agents['coordination_index'].mean():.3f}")
        print(f"Average congestion ratio: {successful_agents['congestion_ratio'].mean():.3f}")

        # Métricas específicas por cenário
        scenario_id = df['scenario_id'].iloc[0]
        if scenario_id in [2, 3]:
            print(f"Average bottleneck time: {successful_agents['bottleneck_time'].mean():.1f} steps")

        if scenario_id in [4, 5]:
            print(f"\n--- COMMUNICATION METRICS ---")
            print(f"Average discovery time: {successful_agents['discovery_steps'].mean():.1f} steps")
            print(f"Average communication efficiency: {successful_agents['communication_efficiency'].mean():.3f}")

    print(f"\n--- ALL AGENTS METRICS ---")
    print(f"Average total collisions: {df['total_collisions'].mean():.1f}")
    print(f"Average safety violations: {df['safety_violations'].mean():.1f}")
    print(f"Average risk exposure time: {df['risk_exposure_time'].mean():.1f} steps")


def evaluate(env_fn, algo="ppo", model_path=None, episodes=10, scenario_id=1, **env_kwargs):
    """Avaliar modelo com métricas completas - incluindo distinção de colisões e métricas secundárias"""
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

        # Métricas básicas
        total_rewards = {agent: 0.0 for agent in agent_names}
        positions = {agent: [] for agent in agent_names}
        distances = {agent: 0.0 for agent in agent_names}
        reached_landmark = {agent: False for agent in agent_names}

        # Métricas de colisão (DISTINTAS)
        collisions_obstacle = {agent: 0 for agent in agent_names}
        collisions_agent = {agent: 0 for agent in agent_names}

        # Métricas de movimento
        stationary_counts = {agent: 0 for agent in agent_names}

        # Métricas de comunicação (cenários 4 e 5)
        discovery_times = {agent: None for agent in agent_names}
        info_sharing_times = {agent: None for agent in agent_names}

        # Métricas de coordenação
        bottleneck_time = {agent: 0 for agent in agent_names}
        near_miss_events = {agent: 0 for agent in agent_names}
        waiting_time = {agent: 0 for agent in agent_names}
        exploration_time = {agent: 0 for agent in agent_names}

        # Métricas finais
        final_rewards = {agent: 0.0 for agent in agent_names}
        evacuation_times = {agent: 0 for agent in agent_names}
        final_distances = {agent: 0.0 for agent in agent_names}

        # Calcular distância inicial
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
        near_miss_threshold = 0.15  # Distância para quase-colisão
        bottleneck_areas = _get_bottleneck_areas(scenario_id)  # Áreas de gargalo por cenário

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

                # Obter objeto do agente
                agent_obj = None
                for a in env.unwrapped.world.agents:
                    if a.name == agent:
                        agent_obj = a
                        break

                # Verificar se alcançou o landmark
                landmark_reached_this_step = False
                if agent_obj and not reached_landmark[agent]:
                    if hasattr(agent_obj, 'goal_rewarded') and agent_obj.goal_rewarded:
                        reached_landmark[agent] = True
                        landmark_reached_this_step = True
                        final_rewards[agent] = total_rewards[agent] + rewards[agent]
                        evacuation_times[agent] = steps
                        final_distances[agent] = distances[agent]
                        print(f"Agent {agent} reached landmark at step {steps}")

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

                    # COLISÕES DISTINTAS
                    if agent_obj:
                        # Colisões com obstáculos e paredes
                        for ob in env.unwrapped.world.obstacles + env.unwrapped.world.walls:
                            if np.linalg.norm(curr_pos - ob.state.p_pos) < (agent_obj.size + ob.size):
                                collisions_obstacle[agent] += 1

                        # Colisões com outros agentes
                        for other_agent in env.unwrapped.world.agents:
                            if other_agent != agent_obj:
                                dist_to_other = np.linalg.norm(curr_pos - other_agent.state.p_pos)
                                if dist_to_other < (agent_obj.size + other_agent.size):
                                    collisions_agent[agent] += 1
                                # Near miss events
                                elif dist_to_other < near_miss_threshold:
                                    near_miss_events[agent] += 1

                    # MÉTRICAS DE COMUNICAÇÃO (cenários 4 e 5)
                    if scenario_id in [4, 5] and agent_obj:
                        # Descoberta do landmark
                        if hasattr(agent_obj, 'discovered_landmark') and agent_obj.discovered_landmark:
                            if discovery_times[agent] is None:
                                discovery_times[agent] = steps

                        # Recebimento de informação de comunicação
                        if hasattr(agent_obj, 'state') and hasattr(agent_obj.state, 'c'):
                            if np.any(agent_obj.state.c != 0) and info_sharing_times[agent] is None:
                                info_sharing_times[agent] = steps

                    # MÉTRICAS DE COORDENAÇÃO
                    # Tempo em bottlenecks (cenários 2 e 3)
                    if scenario_id in [2, 3] and _is_in_bottleneck(curr_pos, bottleneck_areas):
                        bottleneck_time[agent] += 1

                    # Tempo de espera (velocidade muito baixa mas não parado)
                    if agent_obj and hasattr(agent_obj, 'state'):
                        speed = np.linalg.norm(agent_obj.state.p_vel)
                        if 0.01 < speed < 0.05:  # Movimento muito lento
                            waiting_time[agent] += 1

                    # Tempo de exploração (cenários 4 e 5 - antes da descoberta)
                    if scenario_id in [4, 5] and discovery_times[agent] is None:
                        exploration_time[agent] += 1

            # Terminar episódio se todos alcançaram o objetivo
            if all(reached_landmark.values()):
                print("All agents have reached the goal. Ending episode early.")
                break

        # Calcular métricas de grupo
        group_success_rate = sum(reached_landmark.values()) / len(agent_names)
        coordination_index = _calculate_coordination_index(positions, agent_names)

        # Finalizar métricas para agentes que não completaram
        for agent in agent_names:
            if not reached_landmark[agent]:
                final_rewards[agent] = total_rewards[agent]
                evacuation_times[agent] = steps
                final_distances[agent] = distances[agent]

            # Calcular métricas derivadas
            path_eff = direct_distances[agent] / final_distances[agent] if final_distances[agent] > 0 else 0
            congestion_ratio = stationary_counts[agent] / evacuation_times[agent] if evacuation_times[agent] > 0 else 0

            # Métricas de comunicação (valores padrão se não aplicável)
            discovery_steps = discovery_times[agent] if discovery_times[agent] is not None else evacuation_times[agent]
            info_sharing_steps = info_sharing_times[agent] if info_sharing_times[agent] is not None else \
            evacuation_times[agent]
            communication_efficiency = (discovery_steps / evacuation_times[agent]) if evacuation_times[agent] > 0 else 0

            # Métricas de segurança
            total_collisions = collisions_obstacle[agent] + collisions_agent[agent]
            safety_violations = near_miss_events[agent] + total_collisions
            risk_exposure_time = bottleneck_time[agent] + waiting_time[agent]

            # Métricas temporais
            time_to_first_movement = 1  # Assumir movimento no primeiro step
            exploration_ratio = exploration_time[agent] / evacuation_times[agent] if evacuation_times[agent] > 0 else 0

            log_data.append({
                # Métricas básicas
                "episode": ep + 1,
                "agent_id": agent,
                "total_reward": final_rewards[agent],
                "success": int(reached_landmark[agent]),
                "evacuation_time": evacuation_times[agent],
                "distance_travelled": final_distances[agent],
                "direct_distance": direct_distances[agent],
                "path_efficiency": path_eff,

                # Colisões distintas
                "collisions_obstacle": collisions_obstacle[agent],
                "collisions_agent": collisions_agent[agent],
                "total_collisions": total_collisions,

                # Métricas de movimento
                "stationary_steps": stationary_counts[agent],
                "congestion_ratio": congestion_ratio,

                # Métricas de comunicação
                "discovery_steps": discovery_steps,
                "info_sharing_steps": info_sharing_steps,
                "communication_efficiency": communication_efficiency,

                # Métricas de coordenação
                "group_success_rate": group_success_rate,
                "coordination_index": coordination_index,
                "bottleneck_time": bottleneck_time[agent],

                # Métricas de segurança
                "near_miss_events": near_miss_events[agent],
                "safety_violations": safety_violations,
                "risk_exposure_time": risk_exposure_time,

                # Métricas temporais
                "time_to_first_movement": time_to_first_movement,
                "exploration_time": exploration_time[agent],
                "exploration_ratio": exploration_ratio,
                "waiting_time": waiting_time[agent],

                # Contexto
                "scenario_id": scenario_id,
                "algorithm": algo,
                "num_obstacles": len(getattr(env.unwrapped.world, 'obstacles', [])),
                "num_exits": len(getattr(env.unwrapped.world, 'exits', [])),
                "num_agents": len(agent_names),
                "initial_position": json.dumps(initial_positions[agent].tolist(), cls=NumpyEncoder),
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

        # Estatísticas detalhadas
        _print_detailed_statistics(df)
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
        train(env_fn, algo=args.algo, total_steps=250000, scenario_id=args.scenario, **config)

    if args.eval:
        evaluate(env_fn, algo=args.algo, episodes=4, scenario_id=args.scenario, **config)