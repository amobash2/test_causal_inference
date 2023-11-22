import os
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import fsspec
import json


class LoadData:
    def __init__(self,
                 logger,
                 config: Dict):
        self._logger = logger
        self._config = config
    
    def load_data(self):
        file_path = self._config["input_path"] +  self._config["input_file_name"]
        print(f"Loading data from local file at {file_path}")
        data = pd.read_csv(file_path)
        return data
    
class LoadVariables:
    def __init__(self,
                 logger,
                 config: Dict):
        self._logger = logger
        self._config = config

    def load_variables(self):
        variables_spec = None
        variables_path = self._config["variables_path"]

        
        with fsspec.open(variables_path, mode = "r", encoding= "utf-8") as f:
            variables_spec = json.load(f)["variables"]

        if self._logger:
            self._logger.info("Variables and variables spec are successfully loaded...")
        variables = [val for spec in variables_spec for kee, val in spec.items() if kee == "name"]
        
        return variables_spec, variables
    

class VariablesSplit:
    def __init__(self,
                 logger,
                 config,
                 variables: List[str]):
        self._logger = logger
        self._config = config
        self._variables_split = self._load_variables_split()
        self._outcome = self._variables_split["outcome"]
        if len(self._variables_split["treatments"]) == 0:
            if self._logger:
                self._logger.info("Adding all variables as treatment due to null treatment record in the variable split file...")
            self._treatments = list(set(variables) - set([self._outcome]))
        else:
            self._treatments = self._variables_split["treatments"]
        self._non_child_nodes = self._variables_split["non_child_nodes"] if "non_child_nodes" in list(self._variables_split.keys()) else []
        self._forced_no_relation = self._variables_split["forced_no_relation"] if "forced_no_relation" in list(self._variables_split.keys()) else []
        self._forced_with_relation = self._variables_split["forced_with_relation"] if "forced_with_relation" in list(self._variables_split.keys()) else []

    @property
    def outcome(self) -> str:
        return self._outcome
    
    @property
    def treatments(self) -> List:
        return self._treatments
    
    def set_treatments(self, updated_treatments: List):
        self._treatments = updated_treatments
    
    @property 
    def non_child_nodes(self) -> List:
        return self._non_child_nodes
    
    @property
    def forced_no_relation(self) -> List[Tuple]:
        return self._forced_no_relation
    
    @property
    def forced_with_relation(self) -> List[Tuple]:
        return self._forced_with_relation

    def _load_variables_split(self):
        variables_split_path = self._config["variables_split_path"]

        with open(variables_split_path, "r") as f:
            variables_split = json.load(f)
        return variables_split
        
class LoadFeatureToNameMap:
    def __init__(self,
                 logger,
                 config: Dict):
        self._logger = logger
        self._config = config
        self._feature_to_name_map = self._load_feature_to_name_map()

    @property
    def feature_to_name_map(self):
        return self._feature_to_name_map

    def _load_feature_to_name_map(self):
        feature_to_name_file = self._config["feature_to_name_path"]

        with open(feature_to_name_file, "r") as f:
            feature_to_name_map = json.load(f)
        if self._logger:
            self._logger.info("Successfully loaded feature to name map...")
        return feature_to_name_map