import pandas as pd
import numpy as np
from typing import Dict, List

from causal_inference.causal_discovery import CausalDiscovery
from causal_inference.causal_graph import CausalGraph
from causal_inference.causal_analysis import CausalAnalysis
from causal_inference.variability_attribution import VariabilityAttribution
from causal_inference.outlier_analysis import OutlierAnalysis


class RunAllAnalysis:
    def __init__(self,
                 run_time,
                 logger,
                 variables,
                 variables_spec,
                 outcome,
                 treatments,
                 non_child_nodes,
                 forced_no_relation,
                 forced_with_relation,
                 feature_to_name_map,
                 config):
        self._run_time = run_time
        self._run_description = "A causal inference run"
        self._logger = logger
        self._variables = variables
        self._variables_spec = variables_spec
        self._outcome = outcome
        self._treatments = treatments
        self._non_child_nodes = non_child_nodes
        self._forced_no_relation = forced_no_relation
        self._forced_with_relation = forced_with_relation
        self._feature_to_name_map = feature_to_name_map
        self._config = config
    
    def _process_config_df(self):
        config_df = []
        for kee, val in self._config.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    config_df.append({"CONFIG_PARAM_LEVEL_1": kee,
                                      "CONFIG_PARAM_LEVEL_2": k,
                                      "CONFIG_VALUE": str(v)})
            else:
                config_df.append({"CONFIG_PARAM_LEVEL_1": kee,
                                  "CONFIG_PARAM_LEVEL_2": None,
                                  "CONFIG_VALUE": str(val)})
        return config_df


    

    def run(self,
            data: pd.DataFrame(),
            output_path: str):
        if self._logger:
            self._logger.info("Start causal discovery...")
        print("Start causal discovery...")
        print("Variables before discovery:", self._variables)
        causal_discovery = CausalDiscovery(self._logger,
                                        data[self._variables],
                                        self._variables_spec,
                                        self._outcome,
                                        self._treatments,
                                        self._non_child_nodes,
                                        self._forced_no_relation,
                                        self._forced_with_relation,
                                        self._feature_to_name_map,
                                        max_epochs = self._config["max_epochs"])
        lightning_module = causal_discovery.lightning_module
        data_module = causal_discovery.data_module
        print("End of causal discovery...")

        if self._logger:
            self._logger.info("Start of processing causal graph...")
        causal_graph = CausalGraph(self._logger,
                                   self._config,
                                   lightning_module.sem_module,
                                   data_module,
                                   self._feature_to_name_map,
                                   output_path = output_path)
        if self._logger:
            self._logger.info("Ensuring there is no cycle in the graph...")
        cycles = causal_graph.find_cycles()
        if len(cycles) > 0:
            print(f"Cycles: {cycles}")
            message = "There are cycles in the graph."
            message += "This toolkit only works with DAGs.\n"
            message += "Please review these cycles and update the input data for constraints to ensure cycles are not generated.\n"
            if self._logger: self._logger.info(message)
            return message
        
        if self._logger:
            self._logger.info("Storing the graph in the output folder...")
        network_edge_df = causal_graph.store_graph()
        network_edge_path = f"{output_path}/network_edge_df.csv"
        network_edge_df.to_csv(network_edge_path, index = None)

       
        if self._logger:
            self._logger.info("Start of causal analysis for estimating treatment effects...")
        causal_analysis = CausalAnalysis(self._logger,
                                        self._config,
                                        lightning_module.sem_module,
                                        data_module,
                                        self._outcome,
                                        self._treatments,
                                        self._feature_to_name_map,
                                        output_path = output_path,
                                        number_samples= self._config["num_samples_causal_analysis"])
        
        estimated_ate = causal_analysis.estimated_ate
        estimated_ate_path = f"{output_path}/estimated_ate_df.csv"
        if self._logger:
            self._logger.info(f"Saving estimated effects at {estimated_ate_path}...")
        causal_analysis.estimated_ate_df.to_csv(estimated_ate_path, index = None)
        

        causal_analysis.bar_plot({k: v[0] for k, v in estimated_ate.items()},
                                 xlabel = 'LOS change attribution in days ',
                                 plot_save_name= 'causal_analysis_estiamted_ate',
                                 uncertainty_attribs = {k: np.array((v[0]-v[1], v[0]+v[1])) for k, v in estimated_ate.items()})
        if self._config["run_variability_attribution"]:
            variability_attr = VariabilityAttribution(self._logger,
                                                      data[self._variables],
                                                      causal_graph.graph,
                                                      self._outcome,
                                                      self._variables,
                                                      self._feature_to_name_map)
            direct_variability_attribution = variability_attr.direct_variability_attribution

            causal_analysis.bar_plot(variability_attr.convert_to_percentage(variability_attr.iccs),
                                    xlabel='Variance attribution in %',
                                    plot_save_name="variance_attr_percent")
            direct_variability_attribution.to_csv(f"{output_path}/direct_variability_attr.csv", index = None)

        if self._config["run_outlier_analysis"]:
            outlier_analysis = OutlierAnalysis(self._logger,
                                               self._config,
                                               data,
                                               causal_graph.graph,
                                               variability_attr.scm,
                                               self._outcome,
                                               self._variables,
                                               self._feature_to_name_map,
                                               output_path,
                                               self._config["date_column"],
                                               "outlier_analysis_outcome_plot",
                                               outlier_percentile= self._config["outlier_percentile"],
                                               number_bootstrap_resamples= self._config["number_bootstrap_resamples"],
                                               n_jobs = self._config["n_jobs"])
            
            causal_analysis.bar_plot(outlier_analysis.median_attribs,
                                     "Outcome change attribution",
                                     "outcome_change_attribution",
                                     outlier_analysis.uncertainty_attribs)
            
            outlier_df = outlier_analysis.outlier_df
            outlier_df.to_csv(f"{output_path}/outlier_analysis_attribution.csv", index = None)
            
        return lightning_module.sem_module, output_path