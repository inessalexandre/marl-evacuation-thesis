from train_evac import train, evaluate, env_fn
import itertools


algos = ["dqn"]
scenarios = [1, 2, 3, 4]
episodes = 2


scenario_configs = {
    1: {
        "total_steps_list": [500000],
        "param_grid": [{}]
    },
    2: {
        "total_steps_list": [500000],
        "param_grid": [
            {"num_agents": 6}
        ]
    },
    3: {
        "total_steps_list": [500000],
        "param_grid": [
            {"num_agents": 6}
        ]
    },
    4: {
        "total_steps_list": [500000],
        "param_grid": [
            {"num_agents": 6}
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