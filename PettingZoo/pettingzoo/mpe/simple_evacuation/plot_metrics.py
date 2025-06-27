import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import glob
import os
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# CONFIGURAÇÃO PROFISSIONAL DE ESTILO
plt.style.use('default')
sns.set_palette("husl")

# Cores profissionais inspiradas no PDF
COLORS = {
    'ppo': '#2E86AB',  # Azul profissional
    'a2c': '#A23B72',  # Rosa/roxo elegante
    'dqn': '#F18F01',  # Laranja vibrante
    'baseline': '#C73E1D'  # Vermelho para baseline
}

SCENARIO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


class ProfessionalThesisPlots:
    def __init__(self, logs_dir="logs", output_dir="thesis_plots_professional"):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configurar estilo matplotlib
        self.setup_matplotlib_style()

    def setup_matplotlib_style(self):
        """Configurar estilo profissional para matplotlib"""
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.axisbelow': True,
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })

    def calculate_moving_average_with_bounds(self, data, window_size):
        """Calcula média móvel com intervalos de confiança elegantes"""
        if len(data) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

        if len(data) < window_size:
            window_size = max(1, len(data))

        # Média móvel
        moving_avg = data.rolling(window=window_size, min_periods=1, center=False).mean()

        # Intervalo de confiança baseado em desvio padrão
        moving_std = data.rolling(window=window_size, min_periods=1, center=False).std()
        moving_std = moving_std.fillna(0)

        # Intervalos mais conservadores para visualização limpa
        upper_bound = moving_avg + 0.5 * moving_std
        lower_bound = moving_avg - 0.5 * moving_std

        return moving_avg, upper_bound, lower_bound

    def create_reward_convergence_analysis(self, df):
        """Gráfico de convergência estilo paper acadêmico"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        scenarios = sorted(df['scenario_id'].unique())
        algorithms = sorted(df['algorithm'].unique())

        fig.suptitle('Convergência de Recompensas por Cenário de Evacuação',
                     fontsize=18, fontweight='bold', y=0.95)

        for i, scenario in enumerate(scenarios[:4]):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            scenario_data = df[df['scenario_id'] == scenario]

            for algo in algorithms:
                algo_data = scenario_data[scenario_data['algorithm'] == algo]
                if not algo_data.empty:
                    episode_rewards = algo_data.groupby('episode')['total_reward'].mean()

                    if len(episode_rewards) == 0:
                        continue

                    window_size = max(3, int(len(episode_rewards) * 0.04))
                    mov_avg, upper_bound, lower_bound = self.calculate_moving_average_with_bounds(
                        episode_rewards, window_size)

                    if len(mov_avg) == 0:
                        continue

                    x_values = mov_avg.index

                    # Plot linha principal com estilo elegante
                    ax.plot(x_values, mov_avg,
                            color=COLORS.get(algo, '#666666'),
                            linewidth=2.5,
                            label=f'{algo.upper()}',
                            alpha=0.9)

                    # Intervalo de confiança com transparência
                    ax.fill_between(x_values, upper_bound, lower_bound,
                                    color=COLORS.get(algo, '#666666'),
                                    alpha=0.15)

            # Formatação elegante do subplot
            ax.set_title(f'Cenário {scenario}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Episódio de Treino', fontsize=12)
            ax.set_ylabel('Recompensa Acumulada', fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True,
                      loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Adicionar anotações se necessário
            if i == 0:  # Primeiro gráfico
                ax.annotate('Fase de Exploração', xy=(5, mov_avg.iloc[4] if len(mov_avg) > 4 else 0),
                            xytext=(15, mov_avg.iloc[4] + 20 if len(mov_avg) > 4 else 20),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                            fontsize=9, color='gray')

        plt.savefig(f"{self.output_dir}/reward_convergence_professional.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_performance_comparison_matrix(self, df):
        """Matriz de comparação de performance estilo paper"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise Comparativa de Performance entre Algoritmos',
                     fontsize=18, fontweight='bold')

        metrics = [
            ('success', 'Taxa de Sucesso (%)', 100),
            ('evacuation_time', 'Tempo de Evacuação (steps)', 1),
            ('path_efficiency', 'Eficiência do Caminho', 1),
            ('coordination_score', 'Score de Coordenação', 1)
        ]

        algorithms = sorted(df['algorithm'].unique())

        for idx, (metric, ylabel, multiplier) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            for algo in algorithms:
                algo_data = df[df['algorithm'] == algo]
                if not algo_data.empty and metric in algo_data.columns:
                    episode_data = algo_data.groupby('episode')[metric].mean() * multiplier

                    if len(episode_data) == 0:
                        continue

                    window_size = max(2, int(len(episode_data) * 0.06))
                    mov_avg, upper_bound, lower_bound = self.calculate_moving_average_with_bounds(
                        episode_data, window_size)

                    if len(mov_avg) == 0:
                        continue

                    x_values = mov_avg.index

                    # Plot com estilo profissional
                    ax.plot(x_values, mov_avg,
                            color=COLORS.get(algo, '#666666'),
                            linewidth=2.5,
                            label=f'{algo.upper()}',
                            marker='o' if len(x_values) < 20 else None,
                            markersize=3,
                            alpha=0.9)

                    ax.fill_between(x_values, upper_bound, lower_bound,
                                    color=COLORS.get(algo, '#666666'),
                                    alpha=0.15)

            ax.set_title(ylabel, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Episódio', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)

            # Adicionar linha de referência se apropriado
            if metric == 'success':
                ax.axhline(y=80, color='red', linestyle='--', alpha=0.7,
                           label='Meta (80%)')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_comparison_matrix.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_scenario_analysis_dashboard(self, df):
        """Dashboard completo de análise por cenário"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle('Dashboard de Análise por Cenário de Evacuação',
                     fontsize=20, fontweight='bold', y=0.95)

        scenarios = sorted(df['scenario_id'].unique())

        # Gráfico 1: Comparação de Taxa de Sucesso
        ax1 = fig.add_subplot(gs[0, :2])
        success_by_scenario = df.groupby(['scenario_id', 'algorithm'])['success'].mean().unstack()
        success_by_scenario.plot(kind='bar', ax=ax1,
                                 color=[COLORS.get(col, '#666666') for col in success_by_scenario.columns])
        ax1.set_title('Taxa de Sucesso por Cenário', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Taxa de Sucesso (%)')
        ax1.set_xlabel('Cenário')
        ax1.legend(title='Algoritmo', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Tempo de Evacuação
        ax2 = fig.add_subplot(gs[0, 2:])
        evac_time = df.groupby(['scenario_id', 'algorithm'])['evacuation_time'].mean().unstack()
        evac_time.plot(kind='bar', ax=ax2, color=[COLORS.get(col, '#666666') for col in evac_time.columns])
        ax2.set_title('Tempo Médio de Evacuação', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tempo (steps)')
        ax2.set_xlabel('Cenário')
        ax2.legend(title='Algoritmo', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Boxplot de Coordenação
        ax3 = fig.add_subplot(gs[1, :2])
        df_coord = df[df['coordination_score'].notna()]
        if not df_coord.empty:
            sns.boxplot(data=df_coord, x='scenario_id', y='coordination_score',
                        hue='algorithm', ax=ax3, palette=COLORS)
            ax3.set_title('Distribuição de Scores de Coordenação', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Score de Coordenação')
            ax3.set_xlabel('Cenário')
            ax3.grid(True, alpha=0.3)

        # Gráfico 4: Eficiência vs Complexidade
        ax4 = fig.add_subplot(gs[1, 2:])
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            if not algo_data.empty:
                ax4.scatter(algo_data['num_agents'], algo_data['path_efficiency'],
                            color=COLORS.get(algo, '#666666'), label=algo.upper(),
                            alpha=0.6, s=50)
        ax4.set_title('Eficiência vs Complexidade do Cenário', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Número de Agentes')
        ax4.set_ylabel('Eficiência do Caminho')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Gráfico 5: Heatmap de Performance
        ax5 = fig.add_subplot(gs[2, :])
        pivot_data = df.groupby(['scenario_id', 'algorithm'])['total_reward'].mean().unstack()
        if not pivot_data.empty:
            sns.heatmap(pivot_data.T, annot=True, fmt='.1f', cmap='RdYlGn',
                        ax=ax5, cbar_kws={'label': 'Recompensa Média'})
            ax5.set_title('Mapa de Calor: Performance por Cenário e Algoritmo',
                          fontsize=14, fontweight='bold')
            ax5.set_xlabel('Cenário')
            ax5.set_ylabel('Algoritmo')

        plt.savefig(f"{self.output_dir}/scenario_analysis_dashboard.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_coordination_emergence_analysis(self, df):
        """Análise detalhada de coordenação emergente"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análise de Comportamentos Emergentes e Coordenação',
                     fontsize=18, fontweight='bold')

        # Gráfico 1: Evolução temporal da coordenação
        ax1 = axes[0, 0]
        scenario_4_data = df[df['scenario_id'] == 4]
        if not scenario_4_data.empty:
            for algo in scenario_4_data['algorithm'].unique():
                algo_data = scenario_4_data[scenario_4_data['algorithm'] == algo]
                if 'coordination_score' in algo_data.columns:
                    coord_evolution = algo_data.groupby('episode')['coordination_score'].mean()
                    if len(coord_evolution) > 0:
                        window_size = max(2, int(len(coord_evolution) * 0.1))
                        mov_avg, upper_bound, lower_bound = self.calculate_moving_average_with_bounds(
                            coord_evolution, window_size)

                        ax1.plot(mov_avg.index, mov_avg,
                                 color=COLORS.get(algo, '#666666'),
                                 linewidth=2.5, label=algo.upper())
                        ax1.fill_between(mov_avg.index, upper_bound, lower_bound,
                                         color=COLORS.get(algo, '#666666'), alpha=0.2)

        ax1.set_title('Evolução da Coordenação\n(Cenário 4)', fontweight='bold')
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Score de Coordenação')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Correlação Coordenação vs Sucesso
        ax2 = axes[0, 1]
        if 'coordination_score' in df.columns and 'success' in df.columns:
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                ax2.scatter(algo_data['coordination_score'], algo_data['success'],
                            color=COLORS.get(algo, '#666666'), label=algo.upper(),
                            alpha=0.6, s=30)
        ax2.set_title('Coordenação vs Taxa de Sucesso', fontweight='bold')
        ax2.set_xlabel('Score de Coordenação')
        ax2.set_ylabel('Sucesso (0/1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Distribuição de tempos de chegada
        ax3 = axes[0, 2]
        if 'arrival_time' in df.columns:
            arrival_data = df['arrival_time'].dropna()
            if not arrival_data.empty:
                ax3.hist(arrival_data, bins=20, alpha=0.7, color='skyblue',
                         edgecolor='black', density=True)
                ax3.axvline(arrival_data.mean(), color='red', linestyle='--',
                            label=f'Média: {arrival_data.mean():.1f}')
        ax3.set_title('Distribuição de Tempos\nde Chegada', fontweight='bold')
        ax3.set_xlabel('Tempo de Chegada (steps)')
        ax3.set_ylabel('Densidade')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Análise de congestionamento
        ax4 = axes[1, 0]
        if 'congestion_ratio' in df.columns:
            for scenario in df['scenario_id'].unique():
                scenario_data = df[df['scenario_id'] == scenario]
                if not scenario_data.empty:
                    congestion = scenario_data['congestion_ratio'].mean()
                    success = scenario_data['success'].mean()
                    ax4.scatter(congestion, success,
                                color=SCENARIO_COLORS[int(scenario - 1) % len(SCENARIO_COLORS)],
                                s=100, label=f'Cenário {scenario}', alpha=0.8)
        ax4.set_title('Congestionamento vs\nTaxa de Sucesso', fontweight='bold')
        ax4.set_xlabel('Rácio de Congestionamento')
        ax4.set_ylabel('Taxa de Sucesso')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Gráfico 5: Eficácia da comunicação
        ax5 = axes[1, 1]
        if 'communication_effectiveness' in df.columns:
            comm_data = df[df['communication_effectiveness'] > 0]
            if not comm_data.empty:
                for algo in comm_data['algorithm'].unique():
                    algo_data = comm_data[comm_data['algorithm'] == algo]
                    comm_evolution = algo_data.groupby('episode')['communication_effectiveness'].mean()
                    if len(comm_evolution) > 0:
                        ax5.plot(comm_evolution.index, comm_evolution,
                                 color=COLORS.get(algo, '#666666'),
                                 linewidth=2, label=algo.upper(), marker='o')
        ax5.set_title('Eficácia da Comunicação\nao Longo do Tempo', fontweight='bold')
        ax5.set_xlabel('Episódio')
        ax5.set_ylabel('Eficácia da Comunicação')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Gráfico 6: Métricas de exploração
        ax6 = axes[1, 2]
        if 'exploration_coverage' in df.columns:
            exploration_by_algo = df.groupby('algorithm')['exploration_coverage'].mean()
            bars = ax6.bar(exploration_by_algo.index, exploration_by_algo.values,
                           color=[COLORS.get(algo, '#666666') for algo in exploration_by_algo.index],
                           alpha=0.8, edgecolor='black')
            ax6.set_title('Cobertura Média\nde Exploração', fontweight='bold')
            ax6.set_ylabel('Células Exploradas')
            ax6.set_xlabel('Algoritmo')
            ax6.grid(True, alpha=0.3, axis='y')

            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{height:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/coordination_emergence_analysis.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_comprehensive_summary_table(self, df):
        """Tabela resumo estilo paper acadêmico"""
        # Calcular estatísticas
        summary_stats = df.groupby(['scenario_id', 'algorithm']).agg({
            'success': ['mean', 'std', 'count'],
            'evacuation_time': ['mean', 'std'],
            'path_efficiency': ['mean', 'std'],
            'coordination_score': ['mean', 'std'],
            'total_reward': ['mean', 'std']
        }).round(3)

        # Salvar CSV detalhado
        summary_stats.to_csv(f"{self.output_dir}/detailed_summary_statistics.csv")

        # Criar visualização da tabela
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        # Preparar dados para visualização
        table_data = []
        headers = ['Cenário', 'Algoritmo', 'Taxa Sucesso (%)', 'Tempo Evacuação (steps)',
                   'Eficiência Caminho', 'Coordenação', 'Recompensa Total', 'N Experimentos']

        for (scenario, algo), row in summary_stats.iterrows():
            table_data.append([
                f"Cenário {scenario}",
                algo.upper(),
                f"{row[('success', 'mean')] * 100:.1f} ± {row[('success', 'std')] * 100:.1f}",
                f"{row[('evacuation_time', 'mean')]:.0f} ± {row[('evacuation_time', 'std')]:.0f}",
                f"{row[('path_efficiency', 'mean')]:.3f} ± {row[('path_efficiency', 'std')]:.3f}",
                f"{row[('coordination_score', 'mean')]:.3f} ± {row[('coordination_score', 'std')]:.3f}",
                f"{row[('total_reward', 'mean')]:.1f} ± {row[('total_reward', 'std')]:.1f}",
                f"{int(row[('success', 'count')])}"
            ])

        # Criar tabela com formatação profissional
        table = ax.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.08, 0.08, 0.12, 0.14, 0.12, 0.12, 0.12, 0.08])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)

        # Estilizar cabeçalho
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternar cores das linhas
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('white')

        plt.title('Resumo Estatístico Completo dos Resultados Experimentais',
                  fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.output_dir}/comprehensive_summary_table.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def generate_all_professional_plots(self):
        """Gera todos os gráficos profissionais"""
        print("🎨 Carregando dados para gráficos profissionais...")
        df = self.load_all_data()

        if df.empty:
            print("❌ Nenhum dado encontrado!")
            return

        print(f"✅ Dados carregados: {len(df)} registos")
        print(f"📊 Colunas disponíveis: {list(df.columns)}")

        try:
            print("🎯 Criando gráfico de convergência profissional...")
            self.create_reward_convergence_analysis(df)

            print("📈 Criando matriz de comparação de performance...")
            self.create_performance_comparison_matrix(df)

            print("🎛️ Criando dashboard de análise por cenário...")
            self.create_scenario_analysis_dashboard(df)

            print("🤝 Criando análise de coordenação emergente...")
            self.create_coordination_emergence_analysis(df)

            print("📋 Criando tabela resumo profissional...")
            self.create_comprehensive_summary_table(df)

            print(f"\n🎉 GRÁFICOS PROFISSIONAIS CRIADOS COM SUCESSO!")
            print(f"📁 Localização: {self.output_dir}/")
            print("📊 Ficheiros gerados:")
            print("   ✨ reward_convergence_professional.png")
            print("   ✨ performance_comparison_matrix.png")
            print("   ✨ scenario_analysis_dashboard.png")
            print("   ✨ coordination_emergence_analysis.png")
            print("   ✨ comprehensive_summary_table.png")
            print("\n💡 Estes gráficos estão prontos para a tua dissertação!")

        except Exception as e:
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()

    def load_all_data(self):
        """Carrega todos os dados CSV"""
        csv_files = glob.glob(f"{self.logs_dir}/*.csv")
        all_data = []

        for file in csv_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ Erro ao carregar {file}: {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


if __name__ == "__main__":
    generator = ProfessionalThesisPlots()
    generator.generate_all_professional_plots()
