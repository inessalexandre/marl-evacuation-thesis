import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# Configuração do estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def moving_average(data, window=10):
    """Calcular média móvel simples"""
    return data.rolling(window=window, min_periods=1).mean()


def load_training_logs(logs_dir="logs"):
    """Carregar logs de treino (se existirem) ou criar dados simulados"""
    csv_files = glob.glob(f"{logs_dir}/*training*.csv")

    if not csv_files:
        print("No training logs found. Creating simulated training data...")
        return create_training_simulation_data()

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_training_simulation_data():
    """Criar dados simulados de treino baseados na tua implementação"""
    np.random.seed(42)

    algorithms = ['PPO', 'A2C', 'DQN']
    scenarios = [1, 2, 3, 4]
    episodes_per_scenario = 100  # Simular 100 episódios de treino

    data = []

    for algo in algorithms:
        for scenario in scenarios:
            # Simular curvas de aprendizagem realistas
            base_performance = {
                'PPO': {'start': 50, 'end': 120, 'noise': 8},
                'A2C': {'start': 40, 'end': 100, 'noise': 12},
                'DQN': {'start': 30, 'end': 90, 'noise': 15}
            }

            # Ajustar dificuldade por cenário
            scenario_difficulty = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.7}

            perf = base_performance[algo]
            difficulty = scenario_difficulty[scenario]

            # Gerar curva de aprendizagem com melhoria gradual
            episodes = np.arange(1, episodes_per_scenario + 1)

            # Curva sigmoidal de aprendizagem
            progress = 1 / (1 + np.exp(-0.08 * (episodes - 40)))
            base_rewards = perf['start'] + (perf['end'] - perf['start']) * progress * difficulty

            # Adicionar ruído realista
            noise = np.random.normal(0, perf['noise'], len(episodes))
            rewards = base_rewards + noise

            # Adicionar alguns episódios com performance muito baixa (falhas)
            failure_episodes = np.random.choice(episodes, size=max(1, len(episodes) // 20), replace=False)
            rewards[failure_episodes - 1] = np.random.uniform(10, 30, len(failure_episodes))

            for ep, reward in zip(episodes, rewards):
                data.append({
                    'episode': ep,
                    'algorithm': algo,
                    'scenario_id': scenario,
                    'total_reward': reward,
                    'step': ep * 2500  # Convertendo episódio para steps (assumindo 2500 steps por episódio)
                })

    return pd.DataFrame(data)


def plot_reward_curves_with_shading(df, save_dir="figures"):
    """Gráfico 1: Curvas de recompensa por algoritmo com média móvel e shading"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    window_size = 10  # Tamanho da janela para média móvel

    for i, scenario in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        scenario_data = df[df['scenario_id'] == scenario]

        for algo in df['algorithm'].unique():
            algo_data = scenario_data[scenario_data['algorithm'] == algo]

            if len(algo_data) == 0:
                continue

            # Ordenar por episódio
            algo_data = algo_data.sort_values('episode')

            # Usar steps em vez de episódios para o eixo X
            steps = algo_data['step'].values
            rewards = algo_data['total_reward'].values

            # Calcular média móvel e desvio padrão
            rewards_series = pd.Series(rewards)
            ma = moving_average(rewards_series, window_size)
            std = rewards_series.rolling(window=window_size, min_periods=1).std()

            # Plotar linha principal (média móvel)
            ax.plot(steps, ma, label=f'{algo}', linewidth=2.5, alpha=0.9)

            # Adicionar shading (desvio padrão)
            ax.fill_between(steps, ma - std, ma + std, alpha=0.2)

            # Opcional: mostrar pontos originais com transparência
            ax.scatter(steps, rewards, alpha=0.1, s=8)

        ax.set_title(f'Scenario {scenario}: Learning Curves with Moving Average',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Algorithm', fontsize=10)

        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # **AQUI ESTÁ A MUDANÇA PRINCIPAL**: Configurar ticks do eixo X
        # Definir ticks personalizados para mostrar steps em formato legível
        ax.set_xticks([0, 25000, 50000, 100000, 150000, 200000, 250000])
        ax.set_xticklabels(['0', '25k', '50k', '100k', '150k', '200k', '250k'])

        # Definir limites do eixo X
        ax.set_xlim(0, 250000)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/reward_curves_with_shading_steps.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithm_convergence_comparison(df, save_dir="figures"):
    """Gráfico 2: Comparação de convergência entre algoritmos com steps no eixo X"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(14, 8))

    window_size = 15

    # Calcular performance média por algoritmo (todos os cenários)
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]

        # Agrupar por step e calcular média entre cenários
        step_means = algo_data.groupby('step')['total_reward'].mean().reset_index()
        step_stds = algo_data.groupby('step')['total_reward'].std().reset_index()

        steps = step_means['step'].values
        rewards = step_means['total_reward'].values
        stds = step_stds['total_reward'].values

        # Média móvel
        rewards_series = pd.Series(rewards)
        ma = moving_average(rewards_series, window_size)

        # Plotar
        plt.plot(steps, ma, label=f'{algo}', linewidth=3, alpha=0.9)
        plt.fill_between(steps, ma - stds / 2, ma + stds / 2, alpha=0.2)

    plt.title('Algorithm Convergence Comparison (All Scenarios)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average Total Reward', fontsize=14)
    plt.legend(title='Algorithm', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Configurar ticks do eixo X
    plt.xticks([0, 50000, 100000, 150000, 200000, 250000],
               ['0', '50k', '100k', '150k', '200k', '250k'])
    plt.xlim(0, 250000)

    # Melhorar aparência
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/algorithm_convergence_comparison_steps.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_reward_analysis_plots():
    """Função principal para gerar todos os gráficos da secção 5.2"""
    print("Loading training data...")
    df = load_training_logs()

    print(f"Loaded {len(df)} training records")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Scenarios: {sorted(df['scenario_id'].unique())}")
    print(f"Step range: {df['step'].min()} - {df['step'].max()}")

    print("\nGenerating reward function analysis plots...")

    # Gerar gráficos específicos da secção 5.2
    plot_reward_curves_with_shading(df)
    plot_algorithm_convergence_comparison(df)

    print("\nReward analysis plots generated successfully!")
    print("Files saved in 'figures/' directory:")
    print("- reward_curves_with_shading_steps.png")
    print("- algorithm_convergence_comparison_steps.png")

    return df


if __name__ == "__main__":
    df = generate_reward_analysis_plots()
