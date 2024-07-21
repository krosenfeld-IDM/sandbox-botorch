"""
https://ax.dev/tutorials/gpei_hartmann_service.html

Even though there are max parallizations, it looks like there are only single calls
"""

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render
import numpy as np


def evaluate(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x**2).sum()), 0.0)}

if __name__ == "__main__":

    ax_client = AxClient()

    ax_client.create_experiment(
        name="hartmann_test_experiment",
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x4",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x5",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x6",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
        ],
        objectives={"hartmann6": ObjectiveProperties(minimize=True)},
        parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
        outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )


    for i in range(25):
        parameterization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameterization))
        # ax_client.generation_strategy.trials_as_df

    print(f"Max parallelism: {ax_client.get_max_parallelism()}")