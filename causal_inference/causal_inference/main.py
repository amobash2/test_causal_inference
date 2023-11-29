import argparse
import numpy as np
import pandas as pd
import os
import time
import torch
from datetime import datetime
from causal_inference.utils import process_config_file, setup_logging
from causal_inference.load_data import ( 
    LoadData,
    LoadVariables,
    VariablesSplit,
    LoadFeatureToNameMap
)
from causal_inference.run_all_analysis import RunAllAnalysis

import warnings
warnings.filterwarnings("ignore")

def process_variables(original_variables, original_variables_spec, data):
    variables = []
    variables_spec = []
    for o in original_variables:
        if o in list(data):
            variables.append(o)
    for o in original_variables_spec:
        if o["name"] in list(data):
            variables_spec.append(o)
    return variables, variables_spec

def mapping_of_running_var(config, var):
    for val in config["mapping_of_run_variables"]:
        for k, v in val.items():
            if k == var:
                return v
    return None


def save_torch_model(torch_model, output_path):
        model_path = f"{output_path}/causal_discovery_model.pt"
        torch.save(torch_model, model_path)

def causal_inference(config):

    start_time = time.time()
    output_path = config["output_path"]
    output_path = output_path + "/"
    if not os.path.exists(output_path):
        print(f"Creating {output_path} output directory...")
        os.makedirs(output_path)

    print("Setting logging module...")
    logger = setup_logging(config)

    if logger: logger.info("Start of loading the query...")
    try:
        if logger: logger.info("Start of loading data and feature to name map...")
        data = LoadData(logger, config).load_data()
        run_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 

        if logger: logger.info("Start of loading variables...")
        original_variables_spec, original_variables = LoadVariables(logger, config).load_variables()
        if logger: logger.info("Processing variables to ensure they exists in the data...")
        variables, variables_spec = process_variables(original_variables, original_variables_spec, data)
        if logger: logger.info("Extracting outcome and treatment variables...")
        variables_split = VariablesSplit(logger, config, variables)
        if logger: logger.info("Extracting feature to name map...")
        feature_to_name_map = LoadFeatureToNameMap(logger, config).feature_to_name_map

        if logger: logger.info(f"Loading data took {round((time.time() - start_time)/60.0, 3)} minutes...")
    except Exception as e:
        if logger: logger.error(f"Hitting an issue in data processing : {e}")
        print(f"Hitting an issue in data processing : {e}")
        return None, None, None
    
    print(variables)
    if logger: logger.info("Converting variables to float type...")
    data[list(set(variables))] = data[list(set(variables))].astype(float)

    try:
        run_cls = RunAllAnalysis(run_time,
                                    logger,
                                    variables,
                                    variables_spec,
                                    variables_split.outcome,
                                    variables_split.treatments,
                                    variables_split.non_child_nodes,
                                    variables_split.forced_no_relation,
                                    variables_split.forced_with_relation,
                                    feature_to_name_map,
                                    config)
        
        print("Running the analysis for the full dataset at once...")
        if logger: logger.info("Running the analysis for the full dataset at once...")
        torch_model, _ = run_cls.run(data, output_path)
        save_torch_model(torch_model, output_path)
    except Exception as e:
        if logger: logger.error(f"Unable to process causal discovery and analysis phase due to: {e}")
        print(f"Unable to process causal discovery and analysis phase due to: {e}")

    print(f"Finished running causal inference pipeline in {round((time.time() - start_time)/60.0, 5)} minutes...")
    return data, variables_spec, variables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main entry to the causal inference python package developed at Providence.")
    parser.add_argument("--config_path", required = True, help = "Path to the config file.")
    
    args = parser.parse_args()

    config = process_config_file(args.config_path)

    data, variables_spec, variables = causal_inference(config)