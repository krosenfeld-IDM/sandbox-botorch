# README

Directory contents:
- `cl_src/`: LASER source code [commit d20bf93](https://github.com/InstituteforDiseaseModeling/laser/commit/d20bf9364e860b8f97a1f63870a0b11bd7926b64)
- `london/`: London region model (London + 30 km radius)
    - `run_service.py`: Launch `ax` platform for calibration
    - `run_simulation.py`: LASER simulation
    - `scenario_london.py`: Simulation initialization
    - `data/`: Needs to be filled with `distances.npy` and `measles.py` available [here](https://github.com/krosenfeld-IDM/laser-technology-comparison/tree/main/EnglandAndWales)

Requirements are in the `.devtainer/requirements.txt`. 

## Initialize

Before you can run the simulation you'll need to generate the data files (a subset of the larger England and Wales problem). After putting `distances.npy` and `measles.py` in the `data/` folder you can run the following command from the `london/` directory:
```
python scenario_london.py
```
which will create `london/data/londondata.sc` and `london/data/londondist.npy` which holds the demograhics and distances respectively.