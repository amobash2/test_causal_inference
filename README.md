# Background
This folder contains the first draft of a Python package that can be used for causal inference analysis using Causica package. This package is independent of the source of the input data and can run causal inference for a dataframe, given correct configurations and variables. 

This package requires a configuration and few variables file to run a causal inference pipeline as follows:
- `config.yml`: This file contains all necessary parameters for running the causal inference pipeline.
  - An OS environment variable should be defined for the current implementation. `local_workspace`, which can be set to the path containing `causal_inference` folder. This variable is used to direct the implementation to the right paths for `data` folder (`variables_spec.json` and `variables_split.json` files).
- `variables_spec.json`: This file contains the ID and type of each variable that should be included in the analysis. Please note that this file is required for causal discovery and causal analysis steps.
- `feature_to_name_map.json`: This file contains the mapping of variables IDs to their names for better readability of results.
- `variables_split.json`: This file contains the IDs of outcome and treatments variables. It also contains some information for the creation of constraints for the causal discovery phase.

# Folder structure
The folder structure for this python package is as follows:
```
__ setup.py
__ data
  |__ input_data
    |__ hotel_booking.csv (Kaggle hotel booking data)
    |__ config.yml
    |__ variables_spec.json
    |__ variables_split.json
    |__ feature_to_name_map.json
__ causal_inference
  |__ main.py 
  |__ example notebook to run this package
  |__ ...
__ requirements.txt
```
This package can be used for:
- Causal discovery (required)
- Estimating causal effects of given treatment variables on an outcome of interest (required)
- Estimating variability attribution of direct causal factors on an outcome of interest (optional)
- Estimating change attribution of causal factors that result in outlier values for the outcome of interest (optional)

# Package creation
This package currently works on Python 3.10.12 and any version < 3.11.

To use this package, a wheel file can be created and installed locally or on the cloud. To create the wheel file, at the root folder where `setup.py` is stored, please run:
```
python -m build --sdist --wheel
```

This command will create a `.whl` file in a `dist` folder at the root folder. To install the package, navigate to `dist` folder and run:
```
python -m pip install <wheel file name> --force-reinstall
```

If the package is not already installed, `--force-reinstall` can be removed from the above command. 

# How to run
To use this package, first please install all required packages by navigating to the root folder and run:
```
python -m pip install -r requirements.txt
```
Then please set up the files in the `./data` folder based on the input data and use the example notebook at `./causal_inference` folder for your reference to learn how to run causal inference using this package. If you have any questions please create an issue on the repo.