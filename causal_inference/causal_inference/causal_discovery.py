import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Union, Tuple
from operator import itemgetter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from tensordict import TensorDict
from causica.datasets.causica_dataset_format import Variable
from causica.distributions import ContinuousNoiseDist
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule
from causica.sem.structural_equation_model import ite
from causica.training.auglag import AugLagLRConfig
import time

class CausalDiscovery:
    def __init__(self,
                 logger,
                 input_data: pd.DataFrame(),
                 variables_spec: List[Dict],
                 outcome: str,
                 treatments: List,
                 non_child_nodes: List,
                 forced_no_relation: List[Tuple],
                 forced_with_relation: List[Tuple],
                 feature_to_name_map: Dict,
                 batch_size: int = 16,
                 normalize: bool = True,
                 na_substitute: Union[float, None] = 0,
                 prior_sparsity_lambda: float = 1.0,
                 init_rho: float = 1.0,
                 init_alpha: float = 0.0,
                 max_epochs: int = 1500,
                 enable_check_pointing: bool = False,
                 lr_inti_dict: Dict = {"vardist": 1e-2, "icgnn": 3e-4, "noise_dist": 3e-3},
                 random_seed: int = 1):
        self._logger = logger
        self._input_data = input_data
        self._variables_spec = variables_spec
        self._variables = self._process_variables() 
        self._outcome = outcome
        self._treatments = treatments
        self._non_child_nodes = non_child_nodes
        self._forced_no_relations = forced_no_relation
        self._forced_with_relations = forced_with_relation
        self._feature_to_name_map = feature_to_name_map
        self._batch_size = batch_size
        self._normalize = normalize
        self._prior_sparsity_lambda = prior_sparsity_lambda
        self._init_rho = init_rho
        self._init_alpha = init_alpha
        self._max_epochs = max_epochs
        self._enable_check_pointing = enable_check_pointing
        self._lr_init_dict = lr_inti_dict
        self._random_seed = random_seed

        if na_substitute is not None:
            self._input_data.fillna(na_substitute, inplace = True)

        start_time = time.time()
        self._data_module, self._num_nodes = self._process_data_module()
        self._constraint_matrix = self._process_constraints()
        self._lightning_module = self._run()
        if self._logger:
            self._logger.info(f"Causal discovery finished in {round( (time.time() - start_time)/60.0, 3)} minutes...")

    @property
    def lightning_module(self):
        return self._lightning_module
    
    @property
    def data_module(self):
        return self._data_module

    def _process_variables(self):
        return [val for spec in self._variables_spec for kee, val in spec.items() if kee == "name"]
    
    def _process_data_module(self):
        data_module = BasicDECIDataModule(self._input_data,
                                          variables= [Variable.from_dict(d) for d in self._variables_spec],
                                          batch_size= self._batch_size,
                                          normalize= self._normalize,               
        )
        num_nodes = len(data_module.dataset_train.keys())
        return data_module, num_nodes
    
    def _find_all_cat_feature_mapped_names(self, feature_original_name):
        feature_mapped_names = [kee for kee, val in self._feature_to_name_map.items() if val.startswith(feature_original_name)]
        return feature_mapped_names
    
    def _process_non_child_nodes_for_cat(self):
        non_child_nodes = []
        for n in self._non_child_nodes:
            non_child_nodes.append(n)
        return non_child_nodes
    
    def _process_constraints(self):
        node_name_to_idx = {key: i for i, key in enumerate(self._data_module.dataset_train.keys())}
        constraint_matrix = np.full((self._num_nodes, self._num_nodes), np.nan, dtype=np.float32)

        outcome_idx = node_name_to_idx[self._outcome]
        constraint_matrix[outcome_idx, :] = 0.0

        updated_non_child_nodes = self._process_non_child_nodes_for_cat()

        if len(updated_non_child_nodes) > 0:
            try:
                non_child_idxs = itemgetter(*updated_non_child_nodes)(node_name_to_idx)
                constraint_matrix[:, non_child_idxs] = 0.0
            except:
                if self._logger:
                    self._logger.info(f"Unable to add {non_child_idxs} to the non child nodes constraint matrix...")
                pass

        for f in self._forced_no_relations:
            try:
                constraint_matrix[node_name_to_idx[f[0]], node_name_to_idx[f[1]]] = 0.0
            except:
                if self._logger:
                    self._logger.info(f"Unable to add {f} to the forced no relation constraint matrix...")
                pass

        for f in self._forced_with_relations:
            try:
                constraint_matrix[node_name_to_idx[f[0]], node_name_to_idx[f[1]]] = 1.0
            except:
                if self._logger:
                    self._logger.info(f"Unable to add {f} to the forced with relation constraint matrix...")
                pass                        
        return constraint_matrix
    
    def _run(self):
        pl.seed_everything(seed = self._random_seed)

        l_mod = DECIModule(noise_dist=ContinuousNoiseDist.GAUSSIAN,
                                prior_sparsity_lambda=self._prior_sparsity_lambda,
                                init_rho=self._init_rho,
                                init_alpha=self._init_alpha,
                                auglag_config=AugLagLRConfig(
                                    lr_init_dict=self._lr_init_dict
                                    # max_inner_steps=3400,
                                    # max_outer_steps=8,
                                    # lr_init_dict={
                                    #     "icgnn": 0.00076,
                                    #     "vardist": 0.0098,
                                    #     "functional_relationships": 3e-4,
                                    #     "noise_dist": 0.0070,
                                    # },
                                ),
                            )

        l_mod.constraint_matrix = torch.tensor(self._constraint_matrix)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs = self._max_epochs,
            fast_dev_run= False,
            callbacks=[TQDMProgressBar(refresh_rate=19)],
            enable_checkpointing= self._enable_check_pointing,
            logger=False
        )

        trainer.fit(l_mod, datamodule= self._data_module)

        return l_mod
