from train_evac import train, evaluate, env_fn
import itertools


algos = ["ppo"]
scenarios = [1, 2, 3]
episodes = 2


scenario_configs = {
    1: {
        "total_steps_list": [20000],
        "param_grid": [{}]
    },
    2: {
        "total_steps_list": [20000],
        "param_grid": [
            {"num_agents": 6},
            {"num_agents": 10}
        ]
    },
    3: {
        "total_steps_list": [20000],
        "param_grid": [
            {"num_agents": 6, "num_obstacles": 2, "num_exits": 2},
            {"num_agents": 8, "num_obstacles": 3, "num_exits": 3},
            {"num_agents": 12, "num_obstacles": 4, "num_exits": 4}
        ]
    },
    4: {
        "total_steps_list": [200000],
        "param_grid": [
            {"num_agents": 6},
            {"num_agents": 8},
            {"num_agents": 12}
        ]
    }
}


def run_all():
    for scenario_id in scenarios:
        config_set = scenario_configs[scenario_id]
        total_steps_list = config_set["total_steps_list"]
        param_grid = config_set["param_grid"]

        for algo, total_steps, param_combo in itertools.product(algos, total_steps_list, param_grid):
            config = {
                "max_cycles": 150,
                **param_combo
            }

            print(f"=== Training {algo.upper()} | Scenario {scenario_id} | Params={param_combo} | Steps={total_steps} ===")
            model_path = train(
                env_fn,
                algo=algo,
                total_steps=total_steps,
                scenario_id=scenario_id,
                **config
            )

            print(f">>> Evaluating {algo.upper()} | Scenario {scenario_id} <<<")
            evaluate(
                env_fn,
                algo=algo,
                model_path=model_path,
                episodes=episodes,
                scenario_id=scenario_id,
                **config
            )


if __name__ == "__main__":
    run_all()