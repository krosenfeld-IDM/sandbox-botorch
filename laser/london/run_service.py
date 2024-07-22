"""
https://ax.dev/tutorials/gpei_hartmann_service.html

Even though there are max parallizations, it looks like there are only single calls
"""

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render
import numpy as np
import matplotlib.pyplot as plt
from run_simulation import main as london_sim

def evaluate(parameterization):
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"laser": (london_sim(parameterization), 0.0)}

if __name__ == "__main__":

    ax_client = AxClient()

    ax_client.create_experiment(
        name="laser_experiment",
        parameters=[
            {
                "name": "c",
                "type": "range",
                "bounds": [1.0, 3.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            },
            {
                "name": "k",
                "type": "range",
                "bounds": [0.05, 5000.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, defaults to False.
            },
        ],
        objectives={"laser": ObjectiveProperties(minimize=False)}
    )

    for i in range(1100):
        parameterization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameterization))

        if not (i % 100):
            print(f"Completed {i} trials")
            ax_client.save_to_json_file(filepath=f'experiment_{i}.json')


    
    print(f"Max parallelism: {ax_client.get_max_parallelism()}")

    best_parameters, values = ax_client.get_best_parameters()
    print("Best parameters:", best_parameters)

    means, covariances = values
    print("Means:", means)