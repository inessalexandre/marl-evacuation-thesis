import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            max_cycles=100,
            continuous_actions=False,
            render_mode=None,
            scenario_id=1
    ):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            scenario_id=scenario_id
        )

        scenario = Scenario()
        world = scenario.make_world(scenario_id)

        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )

        self.metadata["name"] = f"evacuation_scenario_{scenario_id}"

    def step(self, action):
        result = super().step(action)
        self.scenario.update_communications(self.world)
        return result


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, scenario_id=1):
        world = World()
        world.dim_c = 2
        world.scenario_id = scenario_id

        # Configuração por cenário
        scenarios_config = {
            1: {"agents": 2, "obstacles": 0, "exits": 0, "landmarks": 1, "hide_landmark": False, "comm": False},
            2: {"agents": 6, "obstacles": 24, "exits": 0, "landmarks": 2, "hide_landmark": False, "comm": False},
            3: {"agents": 6, "obstacles": 28, "exits": 2, "landmarks": 2, "hide_landmark": False, "comm": False},
            4: {"agents": 6, "obstacles": 0, "exits": 0, "landmarks": 1, "hide_landmark": True, "comm": True},
        }

        config = scenarios_config.get(scenario_id, scenarios_config[1])
        world.hide_landmark = config["hide_landmark"]
        world.enable_comm = config["comm"]

        # Criar agentes
        world.agents = [Agent() for _ in range(config["agents"])]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = not config["comm"]
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0
            agent.color = np.array([0.15, 0.25, 0.85])

        # Criar landmarks
        world.landmarks = [Landmark() for _ in range(config["landmarks"])]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = False
            lm.movable = False
            lm.size = 0.08
            lm.color = np.array([0.2, 0.9, 0.2])

        # Criar saídas
        world.exits = [Landmark() for _ in range(config["exits"])]
        for i, ex in enumerate(world.exits):
            ex.name = f"exit_{i}"
            ex.collide = False
            ex.movable = False
            ex.size = 0.15
            ex.color = np.array([0.4, 0.7, 1.0])

        # Criar obstáculos
        world.obstacles = [Landmark() for _ in range(config["obstacles"])]
        for i, ob in enumerate(world.obstacles):
            ob.name = f"obstacle_{i}"
            ob.collide = True
            ob.movable = False
            ob.size = 0.1
            ob.color = np.array([0.3, 0.3, 0.3])

        # Adicionar exits e obstacles aos landmarks para renderização
        world.landmarks += world.exits + world.obstacles

        return world

    def reset_world(self, world, np_random):
        # Inicializar estados dos agentes
        for agent in world.agents:
            agent.state.p_pos = np.zeros(2)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(world.dim_c)
            agent.distance_traveled = 0.0
            agent.goal_rewarded = False
            agent.passed_exit = False
            agent.discovered_landmark = False

        # Inicializar landmarks, exits e obstacles
        for entity in world.landmarks:
            entity.state.p_pos = np.zeros(2)
            entity.state.p_vel = np.zeros(2)

        # Configurar posições por cenário
        self._setup_scenario_positions(world, np_random)

        # Definir posições anteriores para cálculo de movimento
        for agent in world.agents:
            agent.last_pos = agent.state.p_pos.copy()

    def _setup_scenario_positions(self, world, np_random):
        """Configurar posições específicas para cada cenário"""

        if world.scenario_id == 1:
            # Cenário 1: Dois agentes, um landmark
            world.landmarks[0].state.p_pos = np.array([0.8, 0.8])
            positions = [np.array([-0.3, -0.8]), np.array([0.3, -0.8])]
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = positions[i] if i < len(positions) else np.array([0.0, -0.8])

        elif world.scenario_id == 2:
            # Cenário 2: Corredor estreito
            world.landmarks[0].state.p_pos = np.array([-0.8, 0.0])
            world.landmarks[1].state.p_pos = np.array([0.8, 0.0])
            positions = [
                np.array([0.0, 0.7]), np.array([-0.1, 0.7]), np.array([0.1, 0.7]),
                np.array([0.0, -0.7]), np.array([-0.1, -0.7]), np.array([0.1, -0.7])
            ]
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = positions[i] if i < len(positions) else np.array([0.0, 0.6])
            self._create_corridor_walls(world)

        elif world.scenario_id == 3:
            # Cenário 3: Obstáculos e saídas
            world.landmarks[0].state.p_pos = np.array([0.0, 0.8])
            world.exits[0].state.p_pos = np.array([-0.6, 0.7])
            world.exits[1].state.p_pos = np.array([0.6, 0.7])
            positions = [
                np.array([-0.7, -0.8]), np.array([-0.5, -0.8]), np.array([0.5, -0.8]),
                np.array([0.7, -0.8]), np.array([-0.3, -0.8]), np.array([0.3, -0.8])
            ]
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = positions[i] if i < len(positions) else np.array([0.0, -0.8])
            self._create_corridor_walls(world)

        elif world.scenario_id == 4:
            # Cenário 4: Comunicação
            world.landmarks[0].state.p_pos = np.array([0.7, 0.7])
            positions = [
                np.array([-0.8, -0.8]), np.array([-0.8, 0.0]), np.array([-0.8, 0.8]),
                np.array([0.0, -0.8]), np.array([0.8, -0.8]), np.array([0.8, 0.0])
            ]
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = positions[i] if i < len(positions) else np.array([-0.8, -0.4 + i * 0.2])

    def _create_corridor_walls(self, world):
        """Criar paredes do corredor para cenário 2 com gaps horizontal e vertical"""
        wall_size = 0.1
        x_gap_size = 1
        y_gap_size = 0.4
        obstacle_idx = 0

        # Paredes verticais
        y_positions = np.arange(-0.9, 0.91, 0.2)
        for x_wall in [-0.5, 0.5]:
            for y_pos in y_positions:
                # Deixa um gap central
                if - y_gap_size / 2 < y_pos < y_gap_size / 2:
                    continue
                if obstacle_idx < len(world.obstacles):
                    world.obstacles[obstacle_idx].state.p_pos = np.array([x_wall, y_pos])
                    #world.obstacles[obstacle_idx].size = wall_size
                    obstacle_idx += 1

        # Paredes horizontais
        x_positions = np.arange(-0.9, 0.91, 0.2)
        for y_wall in [-0.2, 0.2]:
            for x_pos in x_positions:
                # Deixa um gap central
                if - x_gap_size / 2 < x_pos < x_gap_size / 2:
                    continue
                if obstacle_idx < len(world.obstacles):
                    world.obstacles[obstacle_idx].state.p_pos = np.array([x_pos, y_wall])
                    print(f"Obstáculo {obstacle_idx}: posição ({x_wall}, {y_pos})")
                    obstacle_idx += 1

    def is_collision(self, e1, e2):
        """Verificar colisão entre duas entidades"""
        delta = e1.state.p_pos - e2.state.p_pos
        dist = np.linalg.norm(delta)
        return dist < (e1.size + e2.size)

    def reward(self, agent, world):
        """Função de reward adaptada aos diferentes cenários"""
        rew = 0.0

        # Penalização base por step (encoraja soluções rápidas)
        rew -= 0.01

        # Penalização por colisão com obstáculos
        for ob in world.obstacles:
            if self.is_collision(agent, ob):
                rew -= 1.0

        # Penalização por ficar parado (ineficiência)
        if hasattr(agent, 'last_pos'):
            step_dist = np.linalg.norm(agent.state.p_pos - agent.last_pos)
            agent.distance_traveled += step_dist
            agent.last_pos = agent.state.p_pos.copy()
            if step_dist < 0.01:
                rew -= 0.05  # Penaliza ficar parado

        # Recompensas específicas por cenário
        rew += self._scenario_specific_reward(agent, world)

        # Recompensa final por alcançar o objetivo (landmark)
        if not agent.goal_rewarded:
            for lm in world.landmarks:
                if self.is_collision(agent, lm):
                    rew += 100.0
                    agent.goal_rewarded = True
                    break

        return rew

    def _scenario_specific_reward(self, agent, world):
        """Recompensas específicas para cada cenário"""
        rew = 0.0

        # Cenários 1 e 2: aproximação ao landmark
        if world.scenario_id in [1, 2]:
            if world.landmarks and not world.hide_landmark:
                landmark = world.landmarks[0]
                curr_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)

                if not hasattr(agent, 'prev_dist_to_landmark'):
                    agent.prev_dist_to_landmark = curr_dist

                if curr_dist < agent.prev_dist_to_landmark:
                    rew += 5.0 * (agent.prev_dist_to_landmark - curr_dist)

                agent.prev_dist_to_landmark = curr_dist

        # Cenário 3: evacuação por saída antes de ir ao landmark
        elif world.scenario_id == 3:
            if not agent.passed_exit and world.exits:
                # Aproximação à saída
                exit_dists = [np.linalg.norm(agent.state.p_pos - ex.state.p_pos) for ex in world.exits]
                min_exit_dist = min(exit_dists)

                if not hasattr(agent, 'prev_exit_dist'):
                    agent.prev_exit_dist = min_exit_dist

                if min_exit_dist < agent.prev_exit_dist:
                    rew += 3.0 * (agent.prev_exit_dist - min_exit_dist)

                agent.prev_exit_dist = min_exit_dist

                # Passar pela saída
                for ex in world.exits:
                    if self.is_collision(agent, ex):
                        rew += 30.0
                        agent.passed_exit = True
                        break

            elif agent.passed_exit and world.landmarks:
                # Fase 2: aproximação ao landmark
                landmark = world.landmarks[0]
                curr_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)

                if not hasattr(agent, 'prev_landmark_dist'):
                    agent.prev_landmark_dist = curr_dist

                if curr_dist < agent.prev_landmark_dist:
                    rew += 5.0 * (agent.prev_landmark_dist - curr_dist)

                agent.prev_landmark_dist = curr_dist

        # Cenário 4: comunicação
        elif world.scenario_id == 4:
            if not agent.discovered_landmark:
                for lm in world.landmarks:
                    if self.is_collision(agent, lm):
                        rew += 50.0
                        agent.discovered_landmark = True
                        agent.state.c[:] = lm.state.p_pos
                        break

            comm_positions = [
                other.state.c for other in world.agents
                if other.discovered_landmark and other is not agent
            ]

            if comm_positions:
                avg_comm_pos = np.mean(comm_positions, axis=0)
                dist_to_comm = np.linalg.norm(agent.state.p_pos - avg_comm_pos)

                if not hasattr(agent, 'prev_comm_dist'):
                    agent.prev_comm_dist = dist_to_comm

                if dist_to_comm < agent.prev_comm_dist:
                    rew += 3.0 * (agent.prev_comm_dist - dist_to_comm)

                agent.prev_comm_dist = dist_to_comm

        return rew

    def observation(self, agent, world):
        """Observação do agente com tamanho fixo"""
        obs_components = []

        # Estado próprio (4 elementos: vel_x, vel_y, pos_x, pos_y)
        obs_components.extend([agent.state.p_vel, agent.state.p_pos])

        # Landmarks principais (se visíveis) - máximo 1 landmark principal
        landmark_obs = np.zeros(2)  # [rel_x, rel_y]
        if not world.hide_landmark and world.landmarks:
            # Pegar apenas o primeiro landmark (não exits nem obstacles)
            main_landmarks = world.landmarks[:len(world.landmarks) - len(world.exits) - len(world.obstacles)]
            if main_landmarks:
                landmark_obs = main_landmarks[0].state.p_pos - agent.state.p_pos
        obs_components.append(landmark_obs)

        # Saídas - máximo 2 saídas (padding com zeros se necessário)
        exit_obs = np.zeros(4)  # 2 saídas * 2 coordenadas cada
        for i, ex in enumerate(world.exits[:2]):  # Máximo 2 saídas
            exit_obs[i * 2:(i + 1) * 2] = ex.state.p_pos - agent.state.p_pos
        obs_components.append(exit_obs)

        # Obstáculos próximos - SEMPRE 10 obstáculos (padding com zeros)
        obstacle_obs = np.zeros(20)  # 10 obstáculos * 2 coordenadas cada
        nearby_obstacles = []
        for ob in world.obstacles:
            dist = np.linalg.norm(ob.state.p_pos - agent.state.p_pos)
            if dist < 0.5:  # Apenas obstáculos próximos
                nearby_obstacles.append(ob.state.p_pos - agent.state.p_pos)

        # Preencher com os obstáculos próximos (máximo 10)
        for i, ob_pos in enumerate(nearby_obstacles[:10]):
            obstacle_obs[i * 2:(i + 1) * 2] = ob_pos
        obs_components.append(obstacle_obs)

        # Outros agentes - máximo 5 outros agentes (para suportar cenários com 6 agentes)
        other_agents_obs = np.zeros(20)  # 5 agentes * 4 elementos cada (pos_rel + vel)
        other_count = 0
        for other in world.agents:
            if other is not agent and other_count < 5:
                other_agents_obs[other_count * 4:(other_count + 1) * 4] = np.concatenate([
                    other.state.p_pos - agent.state.p_pos,
                    other.state.p_vel
                ])
                other_count += 1
        obs_components.append(other_agents_obs)

        # Comunicação - SEMPRE 2 elementos (padding com zeros se não há comunicação)
        comm_obs = np.zeros(2)
        if world.enable_comm:
            # Média das comunicações de outros agentes que descobriram o landmark
            comm_positions = [
                other.state.c for other in world.agents
                if hasattr(other, 'discovered_landmark') and other.discovered_landmark and other is not agent
            ]
            if comm_positions:
                comm_obs = np.mean(comm_positions, axis=0)
        obs_components.append(comm_obs)

        # Concatenar todas as observações
        # Total: 4 + 2 + 4 + 20 + 20 + 2 = 52 elementos sempre
        return np.concatenate(obs_components)

    def update_communications(self, world):
        """Atualizar comunicações"""
        if world.enable_comm:
            for agent in world.agents:
                if not agent.discovered_landmark:
                    for lm in world.landmarks:
                        if self.is_collision(agent, lm):
                            agent.discovered_landmark = True
                            agent.state.c[:] = lm.state.p_pos
                            break

    def agent_done(self, agent, world):
        """Verificar se agente terminou"""
        if world.scenario_id == 3:
            return agent.passed_exit and agent.goal_rewarded
        else:
            return agent.goal_rewarded
