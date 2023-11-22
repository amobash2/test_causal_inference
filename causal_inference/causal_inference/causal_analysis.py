from typing import Dict, List
import torch
from tensordict import TensorDict
import numpy as np
import pandas as pd
from causica.sem.sem_distribution import SEMDistributionModule
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class CausalAnalysis:
    def __init__(self,
                 logger,
                 config: Dict,
                 sem_module: SEMDistributionModule,
                 data_module,
                 outcome: str,
                 treatments: List,
                 feature_to_name_map: Dict,
                 output_path: str,
                 number_samples: int = 100):
        self._logger = logger
        self._config = config
        self._feature_to_name_map = feature_to_name_map
        self._output_path = output_path
        estimated_ate = {}
        sample_shape = torch.Size([number_samples])

        start_time = time.time()

        sem = sem_module().mode
        transform = data_module.normalizer.transform_modules[outcome]().inv

        for t in treatments:
            intervention_a = TensorDict({t: torch.tensor([1.0])}, batch_size=tuple())
            intervention_b = TensorDict({t: torch.tensor([0.0])}, batch_size=tuple())

            a_samples = transform(sem.do(interventions=intervention_a).sample(sample_shape)[outcome])
            b_samples = transform(sem.do(interventions=intervention_b).sample(sample_shape)[outcome])

            ate_mean = a_samples.mean(0) - b_samples.mean(0)
            ate_std = np.sqrt((a_samples.var(0) + b_samples.var(0)) / number_samples)

            estimated_ate[t] = (
                ate_mean.cpu().numpy()[0],
                ate_std.cpu().numpy()[0]
            )
        
        self._estimated_ate = estimated_ate
        if self._logger:
            self._logger.info(f"Causal analysis finished in {round( (time.time() - start_time)/60.0,3)} minutes...")

    @property
    def estimated_ate_df(self):
        return self._process_estimated_ate(self._estimated_ate)
    
    @property
    def estimated_ate(self):
        return self._estimated_ate
    
    def _process_estimated_ate(self, estimated_ate):
        estimated_ate_df = []
        for k, v in estimated_ate.items():
            estimated_ate_df.append({"FEATURE": k,
                                     "FEATURE_NAME": self._feature_to_name_map[k],
                                     "ESTIMAND": v[0],
                                     "ESTIMAND_LB": v[0] - v[1],
                                     "ESTIMAND_UB": v[0] + v[1]})
        return pd.DataFrame(estimated_ate_df)
    
    def bar_plot(self, value_dictionary, xlabel, plot_save_name, uncertainty_attribs=None, figsize=(12, 7)):
        value_dictionary = {k: v for k, v in sorted(value_dictionary.items(), key=lambda item: item[1], reverse=False)}
        if uncertainty_attribs is None:
            uncertainty_attribs = {node: [value_dictionary[node], value_dictionary[node]] for node in value_dictionary}

        _, ax = plt.subplots(figsize=figsize)
        ci_plus = [uncertainty_attribs[node][1] - value_dictionary[node] for node in value_dictionary.keys()]
        ci_minus = [value_dictionary[node] - uncertainty_attribs[node][0] for node in value_dictionary.keys()]
        xerr = np.array([ci_minus, ci_plus])
        xerr[abs(xerr) < 10**-7] = 0
        for x in list(xerr):
            if x[0] < 0 or x[1] < 0:
                print("Observing negative x: ", x)
        xerr[xerr < 10**-7] = 0 # Due to the matplotlib requirements to have all positive errors
        plt.barh(list(value_dictionary.keys()), value_dictionary.values(), xerr=xerr, ecolor='#1E88E5', color='#ff0d57')#, width=0.8)
        plt.xlabel(xlabel)
        tickpositions = [d for d in list(value_dictionary.keys())]
        plt.yticks(tickpositions, [self._feature_to_name_map[d] for d in list(value_dictionary.keys())], rotation=0, fontsize=6)
        print(value_dictionary.keys())
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        plt.savefig(f"{self._output_path}/{plot_save_name}.jpg", format = "JPG", bbox_inches='tight') 
        plt.close()
        