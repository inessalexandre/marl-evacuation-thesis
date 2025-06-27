from stable_baselines3 import PPO
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss

# Configuração do ambiente
env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=2,
    max_cycles=100,
    continuous_actions=False
)

# Conversão para formato compatível com SB3
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

# Criação do modelo PPO
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

# Carregamento e teste do modelo
model = PPO.load("ppo_tag_model")
env = simple_tag_v3.env(render_mode="human")

obs, _ = env.reset()
for agent in env.agent_iter():
    action, _ = model.predict(obs[agent])
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        break
env.close()
