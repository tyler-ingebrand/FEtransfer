# Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces

Please see the [project page](https://tyler-ingebrand.github.io/FEtransfer/) for more information on this project. 

In this project we compare function encoders against SOTA baselines such as meta learning and transformers. This repo includes the four datasets in the paper, along with the baselines. The function encoder algorithm code is in a pip package [FunctionEncoder](https://github.com/tyler-ingebrand/FunctionEncoder). If this work, or function encoders more generally, are used in your work, please cite:
```
@article{ingebrand_2025_fe_transfer,
  author       = {Tyler Ingebrand and
                  Adam J. Thorpe and
                  Ufuk Topcu},
  title        = {Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces},
  year         = {2025}
}
```

## Installation
```commandline
pip install torch torchvision torchaudio 
pip install FunctionEncoder==0.1.0 numpy matplotlib tqdm tensorboard
pip install gymnasium[mujoco]
```
## Run Experiments
```commandline
./run_all.sh # Runs all experiments and ablations.
```
You may have to give permission to the shell script first, depending on your OS. The shell scripts assume you have 5 GPUs in your machine. You can change this setting at the top of each shell script in ./scripts. 
This script will take a considerable amount of time to run, so you can instead run only the main experiments:
```commandline
./scripts/gather_data.sh
./scripts/run_experiment.sh
```
Alternatively, you can run individual algorithms and datasets using "python test.py" along with various command line arguments. 
Then you can plot using the various scripts in /plots. The plotting scripts require you to first collect the tensorboards, then to run a plotting script. For example, to plot the main experiment, do:
```commandline
python plots/collect_tensorboards.py
python plots/plot_results.py
```
