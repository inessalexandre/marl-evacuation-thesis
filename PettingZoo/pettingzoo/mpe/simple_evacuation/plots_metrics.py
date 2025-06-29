import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime

# Configuração do estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def moving_average(data, window=10):
    """Calcular média móvel simples"""
    return data.rolling(window=window, min_periods=1).mean()


def load_evaluation_data(logs_dir="logs"):
    """Carregar todos os CSVs de avaliação"""
    csv_files = glob.glob(f"{logs_dir}/*.csv")

    if not csv_files:
        print("No CSV files found. Creating sample data for demonstration.")
        return create_sample_data()

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_sample_data():
    """Criar dados de exemplo baseados na estrutura do teu código"""
    np.random.seed(42)
    algorithms = ['PPO', 'A2C', 'DQN']
    scenarios = [1, 2, 3, 4]

    data = []
    for algo in algorithms:
        for scenario in scenarios:
            for episode in range(10):
                # Simular diferentes níveis de dificuldade por cenário
                base_reward = 100 - (scenario - 1) * 15
                base_time = 40 + (scenario - 1) * 10

                # PPO geralmente melhor performance
                if algo == 'PPO':
                    reward_mult = 1.2
                    time_mult = 0.9
                    success_prob = 0.9 - (scenario - 1) * 0.05
                elif algo == 'A2C':
                    reward_mult = 1.0
                    time_mult = 1.0
                    success_prob = 0.85 - (scenario - 1) * 0.08
                else:  # DQN
                    reward_mult = 0.8
                    time_mult = 1.1
                    success_prob = 0.75 - (scenario - 1) * 0.1

                for agent_id in range(2 if scenario == 1 else 6):
                    data.append({
                        'episode': episode + 1,
                        'agent_id': f'agent_{agent_id}',
                        'algorithm': algo,
                        'scenario_id': scenario,
                        'total_reward': np.random.normal(base_reward * reward_mult, 8),
                        'success': np.random.binomial(1, success_prob),
                        'evacuation_time': np.random.normal(base_time * time_mult, 5),
                        'distance_travelled': np.random.uniform(1.5, 4.0),
                        'path_efficiency': np.random.uniform(0.5, 0.95),
                        'collisions': np.random.poisson(scenario * 0.4),
                        'congestion_ratio': np.random.uniform(0.05, 0.35),
                        'num_agents': 2 if scenario == 1 else 6
                    })

    return pd.DataFrame(data)


def plot_success_rate_analysis(df, save_dir="figures"):
    """Gráfico 1: Análise de taxa de sucesso com média móvel e shading"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    window_size = 3

    for i, scenario in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        scenario_data = df[df['scenario_id'] == scenario]

        for algo in df['algorithm'].unique():
            algo_data = scenario_data[scenario_data['algorithm'] == algo]

            if len(algo_data) == 0:
                continue

            # Agrupar por episódio e calcular taxa de sucesso
            success_by_episode = algo_data.groupby('episode')['success'].agg(['mean', 'std']).reset_index()

            # Calcular média móvel
            success_by_episode['mean_ma'] = moving_average(success_by_episode['mean'], window_size)
            success_by_episode['std_ma'] = moving_average(success_by_episode['std'].fillna(0), window_size)

            episodes = success_by_episode['episode'].values
            mean_ma = success_by_episode['mean_ma'].values
            std_ma = success_by_episode['std_ma'].values

            # Plotar linha principal
            ax.plot(episodes, mean_ma, label=f'{algo}', linewidth=2.5, alpha=0.9)

            # Adicionar shading
            ax.fill_between(episodes,
                            np.maximum(0, mean_ma - std_ma),
                            np.minimum(1, mean_ma + std_ma),
                            alpha=0.2)

            # Pontos originais com transparência
            ax.scatter(episodes, success_by_episode['mean'], alpha=0.3, s=20)

        ax.set_title(f'Scenario {scenario}: Success Rate Analysis',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Algorithm', fontsize=10)

        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/success_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_evacuation_time_analysis(df, save_dir="figures"):
    """Gráfico 2: Análise de tempo de evacuação com média móvel e shading"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    window_size = 3

    for i, scenario in enumerate([1, 2, 3, 4]):
        ax = axes[i]
        scenario_data = df[df['scenario_id'] == scenario]

        for algo in df['algorithm'].unique():
            algo_data = scenario_data[scenario_data['algorithm'] == algo]

            if len(algo_data) == 0:
                continue

            # Agrupar por episódio e calcular tempo médio
            time_by_episode = algo_data.groupby('episode')['evacuation_time'].agg(['mean', 'std']).reset_index()

            # Calcular média móvel
            time_by_episode['mean_ma'] = moving_average(time_by_episode['mean'], window_size)
            time_by_episode['std_ma'] = moving_average(time_by_episode['std'].fillna(0), window_size)

            episodes = time_by_episode['episode'].values
            mean_ma = time_by_episode['mean_ma'].values
            std_ma = time_by_episode['std_ma'].values

            # Plotar linha principal
            ax.plot(episodes, mean_ma, label=f'{algo}', linewidth=2.5, alpha=0.9)

            # Adicionar shading
            ax.fill_between(episodes, mean_ma - std_ma, mean_ma + std_ma, alpha=0.2)

            # Pontos originais com transparência
            ax.scatter(episodes, time_by_episode['mean'], alpha=0.3, s=20)

        ax.set_title(f'Scenario {scenario}: Evacuation Time Analysis',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Evacuation Time (steps)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Algorithm', fontsize=10)

        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/evacuation_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_path_efficiency_collision_analysis(df, save_dir="figures"):
    """Gráfico 3: Análise de eficiência de caminho e colisões"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Eficiência do caminho por cenário
    scenario_efficiency = df.groupby(['algorithm', 'scenario_id'])['path_efficiency'].agg(['mean', 'std']).reset_index()

    for algo in df['algorithm'].unique():
        algo_data = scenario_efficiency[scenario_efficiency['algorithm'] == algo]
        scenarios = algo_data['scenario_id'].values
        means = algo_data['mean'].values
        stds = algo_data['std'].values

        axes[0].plot(scenarios, means, marker='o', label=f'{algo}', linewidth=2.5, markersize=8)
        axes[0].fill_between(scenarios, means - stds, means + stds, alpha=0.2)

    axes[0].set_title('Path Efficiency by Algorithm and Scenario', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Scenario ID', fontsize=12)
    axes[0].set_ylabel('Path Efficiency', fontsize=12)
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title='Algorithm')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Gráfico 2: Colisões por cenário
    scenario_collisions = df.groupby(['algorithm', 'scenario_id'])['collisions'].agg(['mean', 'std']).reset_index()

    for algo in df['algorithm'].unique():
        algo_data = scenario_collisions[scenario_collisions['algorithm'] == algo]
        scenarios = algo_data['scenario_id'].values
        means = algo_data['mean'].values
        stds = algo_data['std'].values

        axes[1].plot(scenarios, means, marker='s', label=f'{algo}', linewidth=2.5, markersize=8)
        axes[1].fill_between(scenarios, means - stds, means + stds, alpha=0.2)

    axes[1].set_title('Collision Frequency by Algorithm and Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Scenario ID', fontsize=12)
    axes[1].set_ylabel('Average Collisions per Episode', fontsize=12)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='Algorithm')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/path_efficiency_collision.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison_boxplots(df, save_dir="figures"):
    """Gráfico 4: Comparação de algoritmos com boxplots"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['success', 'evacuation_time', 'path_efficiency', 'collisions']
    titles = ['Success Rate Distribution', 'Evacuation Time Distribution',
              'Path Efficiency Distribution', 'Collision Distribution']
    ylabels = ['Success Rate', 'Evacuation Time (steps)', 'Path Efficiency', 'Number of Collisions']

    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Boxplot com hue para cenários
        sns.boxplot(data=df, x='algorithm', y=metric, hue='scenario_id', ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Melhorar aparência
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/algorithm_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_summary_table(df):
    """Criar tabela resumo de performance para LaTeX"""
    summary = df.groupby(['algorithm', 'scenario_id']).agg({
        'success': ['mean', 'std'],
        'evacuation_time': ['mean', 'std'],
        'path_efficiency': ['mean', 'std'],
        'collisions': ['mean', 'std']
    }).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    print("Performance Summary Table (LaTeX format):")
    print("=" * 100)

    # Criar formato LaTeX
    latex_table = []
    latex_table.append("\\begin{table}[H]")
    latex_table.append("\\renewcommand{\\arraystretch}{1.3}")
    latex_table.append("\\rowcolors{2}{lightgray!30}{white}")
    latex_table.append("\\centering")
    latex_table.append("\\begin{tabularx}{\\textwidth}{@{}p{2cm} p{1.5cm} X X X X@{}}")
    latex_table.append("\\rowcolor{lightgray!50}")
    latex_table.append("\\toprule")
    latex_table.append(
        "\\textbf{Algorithm} & \\textbf{Scenario} & \\textbf{Success Rate (\\%)} & \\textbf{Evac. Time (steps)} & \\textbf{Path Efficiency} & \\textbf{Collisions/Episode} \\\\")
    latex_table.append("\\midrule")

    current_algo = None
    for _, row in summary.iterrows():
        algo = row['algorithm']
        scenario = int(row['scenario_id'])

        if algo != current_algo:
            if current_algo is not None:
                latex_table.append("\\midrule")
            current_algo = algo
            first_row = True
        else:
            first_row = False

        success_mean = row['success_mean'] * 100  # Convert to percentage
        success_std = row['success_std'] * 100
        time_mean = row['evacuation_time_mean']
        time_std = row['evacuation_time_std']
        eff_mean = row['path_efficiency_mean']
        eff_std = row['path_efficiency_std']
        coll_mean = row['collisions_mean']
        coll_std = row['collisions_std']

        if first_row:
            algo_cell = f"\\multirow{{4}}{{*}}{{{algo}}}"
        else:
            algo_cell = ""

        latex_table.append(
            f"{algo_cell} & {scenario} & {success_mean:.1f} ± {success_std:.1f} & {time_mean:.1f} ± {time_std:.1f} & {eff_mean:.2f} ± {eff_std:.2f} & {coll_mean:.1f} ± {coll_std:.1f} \\\\")

    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabularx}")
    latex_table.append(
        "\\caption{Comprehensive performance comparison across algorithms and scenarios. Values represent mean ± standard deviation over evaluation episodes.}")
    latex_table.append("\\label{tab:performance_summary}")
    latex_table.append("\\end{table}")

    for line in latex_table:
        print(line)

    return summary


def generate_all_performance_plots():
    """Função principal para gerar todos os gráficos da secção 5.3"""
    print("Loading evaluation data...")
    df = load_evaluation_data()

    print(f"Loaded {len(df)} records from evaluation logs")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Scenarios: {sorted(df['scenario_id'].unique())}")

    print("\nGenerating performance metrics analysis plots...")

    # Gerar todos os gráficos
    plot_success_rate_analysis(df)
    plot_evacuation_time_analysis(df)
    plot_path_efficiency_collision_analysis(df)
    plot_algorithm_comparison_boxplots(df)

    # Criar tabela resumo
    summary_table = create_performance_summary_table(df)

    print("\nAll performance analysis plots generated successfully!")
    print("Files saved in 'figures/' directory:")
    print("- success_rate_analysis.png")
    print("- evacuation_time_analysis.png")
    print("- path_efficiency_collision.png")
    print("- algorithm_comparison_boxplots.png")

    return df, summary_table


if __name__ == "__main__":
    df, summary = generate_all_performance_plots()
