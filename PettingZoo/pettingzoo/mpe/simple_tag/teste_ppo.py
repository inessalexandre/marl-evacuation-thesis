from stable_baselines3 import PPO
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss

# Configuração do ambiente
env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=2,
    max_cycles=100,
    continuous_actions=False,
    render_mode="human"
)

# Pré-processamento necessário
env = ss.pad_observations_v0(env)  # Garante observações de tamanho fixo
env = ss.pad_action_spac_v0(env)  # Garante espaços de ação consistentes
env = ss.pettingzoo_env_to_vec_env_v1(env)  # Converte para VecEnv
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4)  # Cria múltiplas instâncias

# Criação do modelo PPO com política adequada
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    tensorboard_log="./ppo_tag_tensorboard/"
)

# Treinamento
model.learn(total_timesteps=1_000_000)
model.save("ppo_tag_model")

# Teste do modelo treinado
env = simple_tag_v3.env(render_mode="human")

model = PPO.load("ppo_tag_model")
observations, infos = env.reset()

while env.agents:
    actions = {}
    for agent in env.agents:
        action, _ = model.predict(observations[agent])
        actions[agent] = action

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if any(terminations.values()) or any(truncations.values()):
        break

env.close()
