import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            num_agents=6,
            num_obstacles=4,
            num_exits=2,
            num_landmarks=1,
            max_cycles=100,
            continuous_actions=False,
            render_mode=None,
            scenario_id=1
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            num_obstacles=num_obstacles,
            num_exits=num_exits,
            num_landmarks=num_landmarks,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            scenario_id=scenario_id
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_obstacles, num_exits, num_landmarks, scenario_id)
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
    def make_world(self, num_agents, num_obstacles, num_exits, num_landmarks, scenario_id=1):
        world = World()
        world.dim_c = 2

        if scenario_id == 1:
            num_agents = 2
            num_obstacles = 0
            num_exits = 0

        elif scenario_id == 2:
            num_obstacles = 1
            num_exits = 0

        elif scenario_id == 3:
            pass

        elif scenario_id == 4:
            num_obstacles = 0
            num_exits = 0
            world.hide_landmark = True

        else:
            world.hide_landmark = False

        enable_comm = scenario_id == 4

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = not enable_comm
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        world.real_landmarks = world.landmarks.copy()
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = False
            lm.movable = False
            lm.size = 0.08
            lm.color = np.array([0.2, 0.9, 0.2])

        world.exits = [Landmark() for _ in range(num_exits)]
        for i, ex in enumerate(world.exits):
            ex.name = f"exit_{i}"
            ex.collide = False
            ex.movable = False
            ex.size = 0.25
            ex.color = np.array([0.4, 0.7, 1.0])

        world.obstacles = [Landmark() for _ in range(num_obstacles)]
        for i, ob in enumerate(world.obstacles):
            ob.name = f"obstacle_{i}"
            ob.collide = True
            ob.movable = False
            ob.size = 0.15
            ob.color = np.array([0.3, 0.3, 0.3])

        world.landmarks += world.exits + world.obstacles
        world.scenario_id = scenario_id
        world.enable_comm = enable_comm
        return world

    def reset_world(self, world, np_random):
        self.agent_passed_exit = {agent.name: False for agent in world.agents}

        for agent in world.agents:
            agent.color = np.array([0.15, 0.25, 0.85])
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.discovered_landmark = False
            agent.initial_pos = agent.state.p_pos.copy()
            agent.prev_pos = agent.state.p_pos.copy()
            agent.distance_traveled = 0.0
            agent.prev_dist_to_landmark = float('inf')
            agent.prev_dist_to_exit = float('inf')
            agent.passed_exit = False
            agent.goal_rewarded = False
            agent.last_pos = agent.state.p_pos.copy()
            agent.initial_pos = agent.state.p_pos.copy()

        for obj in world.exits + world.obstacles:
            obj.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            obj.state.p_vel = np.zeros(world.dim_p)

        fixed_positions = [np.array([-0.8, 0.8]), np.array([0.8, -0.8])]
        for i, lm in enumerate(world.landmarks[:1]):
            lm.state.p_pos = fixed_positions[i % len(fixed_positions)]
            lm.state.p_vel = np.zeros(world.dim_p)

        if world.scenario_id == 2:
            # Landmark fixa no canto superior direito
            world.landmarks[0].state.p_pos = np.array([0.8, 0.8])
            world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

            for i, agent in enumerate(world.agents):
                agent.state.p_pos = np.array([-0.8, -0.2 + i * 0.4])
                agent.state.p_vel = np.zeros(world.dim_p)

            if len(world.obstacles) >= 1:
                world.obstacles[0].state.p_pos = np.array([0.0, 0.0])
                world.obstacles[0].state.p_vel = np.zeros(world.dim_p)
                world.obstacles[0].size = 0.1

    def is_collision(self, e1, e2):
        delta = e1.state.p_pos - e2.state.p_pos
        dist = np.linalg.norm(delta)
        return dist < (e1.size + e2.size)

    def reward(self, agent, world):
        # Initialize agent tracking variables
        if not hasattr(agent, 'last_pos'):
            agent.last_pos = agent.state.p_pos.copy()
            agent.distance_traveled = 0.0
        if not hasattr(agent, 'prev_dist_to_landmark'):
            agent.prev_dist_to_landmark = float('inf')
        if not hasattr(agent, 'goal_rewarded'):
            agent.goal_rewarded = False

        # Update distance traveled
        step_dist = np.linalg.norm(agent.state.p_pos - agent.last_pos)
        agent.distance_traveled += step_dist
        agent.last_pos = agent.state.p_pos.copy()

        rew = -0.3  # Step penalty

        # Obstacle collision penalty
        for ob in world.obstacles:
            if self.is_collision(agent, ob):
                rew -= 2.0

        # Landmark approach rewards (applicable in all scenarios)
        landmark_rewards_active = True
        if world.scenario_id == 3:
            landmark_rewards_active = agent.passed_exit
        if world.scenario_id == 4:
            landmark_rewards_active = False

        if landmark_rewards_active and world.landmarks and not getattr(world, "hide_landmark", False):
            landmark = world.landmarks[0]
            curr_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)

            if agent.prev_dist_to_landmark == float('inf'):
                agent.prev_dist_to_landmark = curr_dist

            dist_reduction = agent.prev_dist_to_landmark - curr_dist
            rew += 0.8 * (dist_reduction ** 2)
            agent.prev_dist_to_landmark = curr_dist

            # Additional incentives
            rew -= 0.005 * agent.distance_traveled
            ideal_dir = (landmark.state.p_pos - agent.state.p_pos)
            ideal_dir /= np.linalg.norm(ideal_dir) + 1e-8
            vel_dir = agent.state.p_vel / (np.linalg.norm(agent.state.p_vel) + 1e-8)
            rew += 0.3 * np.dot(ideal_dir, vel_dir)

            if agent.distance_traveled > 0:
                direct_dist = np.linalg.norm(agent.initial_pos - landmark.state.p_pos)
                path_efficiency = direct_dist / agent.distance_traveled
                rew += 0.15 * path_efficiency

        if world.scenario_id == 3 and world.exits:
            if not agent.passed_exit:
                exit_dists = [np.linalg.norm(agent.state.p_pos - ex.state.p_pos)
                              for ex in world.exits]
                min_exit_dist = min(exit_dists)

                if not hasattr(agent, 'prev_exit_dist'):
                    agent.prev_exit_dist = min_exit_dist

                if min_exit_dist < agent.prev_exit_dist:
                    rew += 1 * (agent.prev_exit_dist - min_exit_dist)
                agent.prev_exit_dist = min_exit_dist

                for ex in world.exits:
                    if self.is_collision(agent, ex):
                        rew += 6.0
                        agent.passed_exit = True
            else:
                for ex in world.exits:
                    if self.is_collision(agent, ex):
                        rew += 6.0
                        agent.passed_exit = True

                if world.landmarks:
                    landmark = world.landmarks[0]
                    landmark_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)

                    if landmark_dist < agent.prev_dist_to_landmark:
                        rew -= 0.3 * (agent.prev_dist_to_landmark - landmark_dist)
                    agent.prev_dist_to_landmark = landmark_dist

                if not agent.goal_rewarded:
                    for lm in world.real_landmarks:
                        if self.is_collision(agent, lm):
                            rew += 20
                            agent.goal_rewarded = True

        if world.scenario_id == 4:
            if not agent.discovered_landmark:
                for lm in world.landmarks:
                    if self.is_collision(agent, lm):
                        rew += 15
                        agent.discovered_landmark = True
                        agent.state.c[:] = lm.state.p_pos

            if any(other.discovered_landmark for other in world.agents):
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
                        rew += 1.5 * (agent.prev_comm_dist - dist_to_comm)

                    agent.prev_comm_dist = dist_to_comm

        # Landmark reward for all scenarios
        if not agent.goal_rewarded:
            for lm in world.landmarks:
                if self.is_collision(agent, lm):
                    rew += 20
                    agent.goal_rewarded = True

        # Collective bonus
        if all(getattr(a, 'goal_rewarded', False) for a in world.agents):
            rew += 10

        # Velocity penalty at completion
        if self.agent_done(agent, world):
            rew -= 2.0 * np.linalg.norm(agent.state.p_vel)

        return rew

    def observation(self, agent, world):
        landmark_pos = []
        if not getattr(world, "hide_landmark", False):
            landmark_pos = [lm.state.p_pos - agent.state.p_pos for lm in world.landmarks]

        exit_pos = [ex.state.p_pos - agent.state.p_pos for ex in world.exits]
        obstacle_pos = [ob.state.p_pos - agent.state.p_pos for ob in world.obstacles]

        other_pos, other_vel, comm = [], [], []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)

            if getattr(world, "enable_comm", False) and not other.silent:
                comm.append(other.state.c)

        obs = [agent.state.p_vel, agent.state.p_pos] + landmark_pos + exit_pos + obstacle_pos + other_pos + other_vel
        if comm:
            obs += comm

        return np.concatenate(obs)

    def update_communications(self, world):
        for agent in world.agents:
            if not agent.discovered_landmark:
                for lm in world.landmarks:
                    if self.is_collision(agent, lm):
                        agent.discovered_landmark = True
                        agent.state.c[:] = lm.state.p_pos
            elif world.scenario_id == 4:
                if world.landmarks:
                    agent.state.c[:] = world.landmarks[0].state.p_pos


    def agent_done(self, agent, world):
        if world.scenario_id == 3:
            return agent.passed_exit and agent.goal_rewarded
        else:
            return agent.goal_rewarded