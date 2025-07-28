import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# CORREÇÃO: Adicionar configuração de estilo igual ao plots_c1.py
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def moving_average(data, window_size):
    """Calcular média móvel"""
    return data.rolling(window=window_size, min_periods=1).mean()


def load_real_training_data(logs_dir="logs"):
    """Carregar dados reais de treino"""
    training_files = glob.glob(f"{logs_dir}/reward_*.csv")
    if not training_files:
        print("Nenhum ficheiro de treino encontrado!")
        print(f"Ficheiros encontrados em {logs_dir}:")
        all_files = glob.glob(f"{logs_dir}/*.csv")
        for f in all_files:
            print(f" - {f}")
        return None

    dfs = []
    for file in training_files:
        try:
            df = pd.read_csv(file)
            print(f"Carregado: {file} com {len(df)} registos")
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total de registos de treino: {len(combined_df)}")
        print(f"Algoritmos: {combined_df['algorithm'].unique()}")
        print(f"Cenários: {combined_df['scenario'].unique()}")
        return combined_df
    return None


def plot_real_reward_curves_with_shading(df, save_dir="figures"):
    """Gráfico com dados reais de treino - CORRIGIDO para estética do plots_c1.py"""
    if df is None:
        print("Sem dados reais para plotar")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    window_size = 15  # CORREÇÃO: Usar mesmo window_size do plots_c1.py

    for i, scenario in enumerate([1,2,3,4,5]):
        ax = axes[i]
        scenario_data = df[df['scenario'] == scenario]

        if len(scenario_data) == 0:
            ax.text(0.5, 0.5, f'No data for Scenario {scenario}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        for algo in scenario_data['algorithm'].unique():
            algo_data = scenario_data[scenario_data['algorithm'] == algo]
            if len(algo_data) == 0:
                continue

            # Ordenar por timestep - CORREÇÃO: Adicionar ordenação
            algo_data = algo_data.sort_values('timestep')

            # Usar dados reais
            steps = algo_data['timestep'].values
            rewards = algo_data['reward'].values

            # Média móvel
            rewards_series = pd.Series(rewards)
            ma = moving_average(rewards_series, window_size)
            std = rewards_series.rolling(window=window_size, min_periods=1).std()

            # CORREÇÃO: Plotar com estilo igual ao plots_c1.py
            ax.plot(steps, ma, label=f'{algo.upper()}', linewidth=2.5, alpha=0.9)
            ax.fill_between(steps, ma - std, ma + std, alpha=0.2)
            ax.scatter(steps, rewards, alpha=0.1, s=8)

        # CORREÇÃO: Títulos e labels com formatação igual ao plots_c1.py
        ax.set_title(f'Scenario {scenario}: Reward - Learning Curve',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Algorithm', fontsize=10)

        # CORREÇÃO: Melhorar aparência - igual ao plots_c1.py
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # CORREÇÃO: Configurar eixo X baseado nos dados reais com formatação melhorada
        max_steps = df['timestep'].max()
        if max_steps > 0:
            if max_steps >= 700000:
                # Se dados chegam a 700k, usar formatação igual ao plots_c1.py
                ax.set_xticks([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000])
                ax.set_xticklabels(['0', '100k', '200k', '300k', '400k', '500k', '600k', '700k'])
                ax.set_xlim(0, 700000)
            else:
                # Caso contrário, adaptar aos dados disponíveis
                step_ticks = np.linspace(0, max_steps, 6).astype(int)
                step_labels = [f'{s // 1000}k' if s >= 1000 else str(s) for s in step_ticks]
                ax.set_xticks(step_ticks)
                ax.set_xticklabels(step_labels)
                ax.set_xlim(0, max_steps)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/real_reward_curves_with_shading.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_real_reward_plots():
    """Função principal para gerar gráficos com dados reais"""
    print("=== GENERATING PLOTS WITH REAL TRAINING DATA ===")

    # Carregar dados reais
    df = load_real_training_data()

    if df is not None:
        print("✓ Dados reais de treino carregados")
        plot_real_reward_curves_with_shading(df)
        print("✓ Gráficos gerados com dados reais!")
    else:
        print("⚠ Nenhum dado real encontrado. Execute primeiro o treino.")
        print("Para gerar dados reais:")
        print("1. Executa: python train_evac_old.py --train --algo ppo --scenario 1")
        print("2. Repete para outros algoritmos e cenários")

    return df


if __name__ == "__main__":
    df = generate_real_reward_plots()
