import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime

# Configuração de estilo igual ao plot_reward.py
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def moving_average(data, window_size):
    """Calcular média móvel"""
    return data.rolling(window=window_size, min_periods=1).mean()


def load_evaluation_data(logs_dir="logs"):
    """Carregar todos os dados de avaliação"""
    eval_files = glob.glob(f"{logs_dir}/*.csv")

    if not eval_files:
        print("Nenhum ficheiro de avaliação encontrado!")
        print(f"Ficheiros encontrados em {logs_dir}:")
        all_files = glob.glob(f"{logs_dir}/*")
        for f in all_files:
            print(f" - {f}")
        return None

    dfs = []
    for file in eval_files:
        try:
            df = pd.read_csv(file)
            df['log_file'] = os.path.basename(file)
            print(f"Carregado: {file} com {len(df)} registos")
            dfs.append(df)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total de registos de avaliação: {len(combined_df)}")
        print(f"Algoritmos: {combined_df['algorithm'].unique()}")
        print(f"Cenários: {combined_df['scenario_id'].unique()}")
        return combined_df

    return None


def get_metrics_categories():
    """Categorizar métricas para melhor organização"""
    return {
        'performance': [
            'total_reward', 'success', 'evacuation_time', 'path_efficiency'
        ],
        'collisions': [
            'collisions_obstacle', 'collisions_agent', 'total_collisions',
            'near_miss_events', 'safety_violations'
        ],
        'movement': [
            'distance_travelled', 'stationary_steps', 'congestion_ratio',
            'waiting_time', 'exploration_time'
        ],
        'coordination': [
            'group_success_rate', 'coordination_index', 'bottleneck_time',
            'risk_exposure_time'
        ],
        'communication': [
            'discovery_steps', 'info_sharing_steps', 'communication_efficiency',
            'exploration_ratio'
        ]
    }


def plot_metric_by_scenario_individual(df, metric, window_size=10, save_dir="figures/evaluation"):
    os.makedirs(save_dir, exist_ok=True)
    scenarios = sorted(df['scenario_id'].unique())
    generated_files = []
    for scenario in scenarios:
        scenario_data = df[df['scenario_id'] == scenario]
        if len(scenario_data) == 0:
            continue
        plt.figure(figsize=(8, 6))
        for algo in sorted(scenario_data['algorithm'].unique()):
            algo_data = scenario_data[scenario_data['algorithm'] == algo]
            if len(algo_data) == 0:
                continue
            episode_data = algo_data.groupby('episode')[metric].agg(['mean', 'std']).reset_index()
            if len(episode_data) < 2:
                continue
            ma = moving_average(episode_data['mean'], window_size)
            std_ma = moving_average(episode_data['std'].fillna(0), window_size)
            episodes = episode_data['episode'].values
            plt.plot(episodes, ma, label=f'{algo.upper()}', linewidth=2.5, alpha=0.9)
            plt.fill_between(episodes, ma - std_ma, ma + std_ma, alpha=0.2)
            plt.scatter(episodes, episode_data['mean'], alpha=0.1, s=8)
        plt.title(f'Scenario {scenario}: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Algorithm', fontsize=10)
        plt.tight_layout()
        filename = f'{save_dir}/{metric}_scenario_{scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
    return generated_files


def plot_comparison_matrix(df, metrics, save_dir="figures/evaluation"):
    """Criar matriz de comparação entre algoritmos e cenários"""
    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics:
        # Criar tabela pivot com média da métrica
        pivot_data = df.groupby(['algorithm', 'scenario_id'])[metric].mean().unstack(fill_value=0)

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': metric.replace("_", " ").title()})
        plt.title(f'{metric.replace("_", " ").title()} - Algorithm vs Scenario Comparison',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Scenario', fontsize=12)
        plt.ylabel('Algorithm', fontsize=12)

        filename = f'{save_dir}/{metric}_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_success_rate_analysis(df, save_dir="figures/evaluation"):
    """Análise específica de taxa de sucesso"""
    os.makedirs(save_dir, exist_ok=True)

    # Taxa de sucesso por algoritmo e cenário
    success_data = df.groupby(['algorithm', 'scenario_id'])['success'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 8))

    scenarios = sorted(df['scenario_id'].unique())
    algorithms = sorted(df['algorithm'].unique())

    x = np.arange(len(scenarios))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        algo_data = success_data[success_data['algorithm'] == algo]
        means = [algo_data[algo_data['scenario_id'] == s]['mean'].iloc[0]
                 if len(algo_data[algo_data['scenario_id'] == s]) > 0 else 0
                 for s in scenarios]
        stds = [algo_data[algo_data['scenario_id'] == s]['std'].iloc[0]
                if len(algo_data[algo_data['scenario_id'] == s]) > 0 else 0
                for s in scenarios]

        plt.bar(x + i * width, means, width, label=algo.upper(),
                yerr=stds, capsize=5, alpha=0.8)

    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate by Algorithm and Scenario', fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(algorithms) - 1) / 2, scenarios)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    filename = f'{save_dir}/success_rate_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def plot_correlation_analysis(df, save_dir="figures/evaluation"):
    """Análise de correlação entre métricas"""
    os.makedirs(save_dir, exist_ok=True)

    # Selecionar métricas numéricas
    numeric_metrics = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['episode', 'scenario_id', 'num_agents', 'num_obstacles', 'num_exits']
    metrics_for_corr = [col for col in numeric_metrics if col not in exclude_cols]

    if len(metrics_for_corr) < 2:
        return None

    # Matriz de correlação
    corr_matrix = df[metrics_for_corr].corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix - Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{save_dir}/correlation_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def generate_all_evaluation_plots(df, save_dir="figures/evaluation"):
    """Gerar todos os gráficos de avaliação"""
    print("=== GENERATING EVALUATION PLOTS ===")

    # Obter todas as métricas disponíveis
    exclude_cols = ['episode', 'agent_id', 'algorithm', 'scenario_id', 'log_file',
                    'initial_position', 'trajectory', 'num_agents', 'num_obstacles',
                    'num_exits']

    all_metrics = [col for col in df.columns if col not in exclude_cols]
    metrics_categories = get_metrics_categories()

    generated_plots = []

    print(f"Métricas disponíveis: {all_metrics}")

    # 1. Gráficos individuais por métrica
    print("Gerando gráficos individuais por métrica...")
    for metric in all_metrics:
        if df[metric].dtype in ['int64', 'float64']:
            try:
                filename = plot_metric_by_scenario_individual(df, metric, save_dir=save_dir)
                generated_plots.append(filename)
                print(f"✓ {metric}")
            except Exception as e:
                print(f"✗ Erro em {metric}: {e}")

    # 2. Matriz de comparação (heatmaps)
    print("Gerando matrizes de comparação...")
    numeric_metrics = [col for col in all_metrics if df[col].dtype in ['int64', 'float64']]
    plot_comparison_matrix(df, numeric_metrics[:10], save_dir=save_dir)  # Limitar a 10 para não sobrecarregar

    # 3. Análise de taxa de sucesso
    print("Gerando análise de taxa de sucesso...")
    if 'success' in df.columns:
        success_plot = plot_success_rate_analysis(df, save_dir=save_dir)
        generated_plots.append(success_plot)

    # 4. Análise de correlação
    print("Gerando análise de correlação...")
    corr_plot = plot_correlation_analysis(df, save_dir=save_dir)
    if corr_plot:
        generated_plots.append(corr_plot)

    # 5. Gráficos por categoria
    print("Gerando gráficos por categoria...")
    for category, metrics in metrics_categories.items():
        available_metrics = [m for m in metrics if m in all_metrics]
        if available_metrics:
            category_dir = f"{save_dir}/{category}"
            os.makedirs(category_dir, exist_ok=True)

            for metric in available_metrics:
                if df[metric].dtype in ['int64', 'float64']:
                    try:
                        filename = plot_metric_by_scenario(df, metric, save_dir=category_dir)
                        generated_plots.append(filename)
                    except Exception as e:
                        print(f"✗ Erro em {category}/{metric}: {e}")

    print(f"✓ Total de gráficos gerados: {len(generated_plots)}")
    return generated_plots


def main():
    """Função principal"""
    print("=== ANÁLISE DE MÉTRICAS DE AVALIAÇÃO ===")

    # Carregar dados
    df = load_evaluation_data()

    if df is None:
        print("⚠ Nenhum dado de avaliação encontrado!")
        print("Para gerar dados de avaliação:")
        print("1. Execute: python train_ecav.py --eval --algo ppo --scenario 1")
        print("2. Repita para outros algoritmos e cenários")
        return

    # Gerar todos os gráficos
    plots = generate_all_evaluation_plots(df)

    print("\n=== RESUMO DOS GRÁFICOS GERADOS ===")
    for plot in plots[:20]:  # Mostrar apenas os primeiros 20
        print(f"- {plot}")

    if len(plots) > 20:
        print(f"... e mais {len(plots) - 20} gráficos")

    print(f"\n✓ Todos os gráficos salvos em: figures/evaluation/")

    # Estatísticas gerais
    print("\n=== ESTATÍSTICAS GERAIS ===")
    print(f"Total de registos: {len(df)}")
    print(f"Algoritmos: {', '.join(df['algorithm'].unique())}")
    print(f"Cenários: {', '.join(map(str, sorted(df['scenario_id'].unique())))}")
    print(f"Episódios por combinação: {df.groupby(['algorithm', 'scenario_id']).size().describe()}")


if __name__ == "__main__":
    main()
