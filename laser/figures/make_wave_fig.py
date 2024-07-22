import os
import sys
sys.path.append(os.path.join("..","london"))

from ax.service.ax_client import AxClient
from run_simulation import main as london_sim


i = 400
ax_client = AxClient.load_from_json_file(filepath=os.path.join("..","london",f"experiment_{i}.json"))

best_parameters, values = ax_client.get_best_parameters()
print("Best parameters:", best_parameters)

means, covariances = values
print("Means:", means)

run_simulation = london_sim(best_parameters, do_plot=True)