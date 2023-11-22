import pandas as pd
import numpy as np
from typing import List, Dict
from dowhy import gcm
import time

class VariabilityAttribution:
    def __init__(self,
                 logger,
                 input_data: pd.DataFrame(),
                 graph,
                 outcome: str,
                 variables: List,
                 feature_to_name_map: Dict,
                 number_samples: int = 100):
        self._logger = logger # TBA more logging here
        self._input_data = input_data
        self._graph = graph
        self._outcome = outcome
        self._variables = variables
        self._feature_to_name_map = feature_to_name_map
        self._number_samples = number_samples
        start_time = time.time()
        self._direct_variability_attribution, self._iccs, self._scm = self._run()
        if self._logger: self._logger.info(f"Variability attribution finished in {round( (time.time() - start_time)/60.0,3)} minutes...")
        print(f"Variability attribution finished in {round( (time.time() - start_time)/60.0,3)} minutes...")

    @property
    def direct_variability_attribution(self):
        return self._direct_variability_attribution

    @property
    def iccs(self):
        return self._iccs
    
    @property
    def scm(self):
        return self._scm

    def convert_to_percentage(self, value_dictionary):
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}
    
    def _run(self):
        scm = gcm.StructuralCausalModel(self._graph)
        gcm_df = self._input_data[self._variables]
        gcm_df.fillna(0, inplace=True)
        gcm.auto.assign_causal_mechanisms(scm, gcm_df)

        gcm.fit(scm, gcm_df)

        arrow_strengths = gcm.arrow_strength(scm, target_node = self._outcome)

        arrow_strengths_df = []
        for kee, val in self.convert_to_percentage(arrow_strengths).items():
            arrow_strengths_df.append({"SOURCE_NAME": self._feature_to_name_map[kee[0]],
                                       "TARGET_NAME": self._feature_to_name_map[kee[1]],
                                       "PERCENTAGE_INFLUENCE_ON_VARIANCE": val})
            
        iccs = gcm.intrinsic_causal_influence(scm, target_node = self._outcome, num_samples_randomization = self._number_samples)
            
        return pd.DataFrame(arrow_strengths_df).sort_values(by="PERCENTAGE_INFLUENCE_ON_VARIANCE", ascending=False), iccs, scm