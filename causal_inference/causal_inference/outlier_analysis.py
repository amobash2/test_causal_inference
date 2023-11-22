import pandas as pd
import numpy as np
from dowhy import gcm
from typing import List, Union, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class OutlierAnalysis:
    def __init__(self,
                 logger,
                 config,
                 input_data: pd.DataFrame(),
                 graph,
                 scm,
                 outcome: str,
                 variables: List,
                 feature_to_name_map: Dict,
                 output_path: str,
                 plot_save_name: str,
                 outlier_percentile: int = 90,
                 number_bootstrap_resamples: int = 1,
                 n_jobs: int = 1):
        self._logger = logger # TBA more logging here
        self._config = config
        self._input_data.fillna(0, inplace=True)
        self._graph = graph
        self._scm = scm
        self._outcome = outcome
        self._variables = variables
        self._feature_to_name_map = feature_to_name_map
        self._output_path = output_path
        self._plot_save_name = plot_save_name
        self._outlier_percentile = outlier_percentile
        self._number_bootstrap_resamples = number_bootstrap_resamples
        self._n_jobs = n_jobs
        start_time = time.time()
        self._median_attribs, self._uncertainty_attribs, self._outlier_df = self._run()
        if self._logger:
            self._logger.info(f"Outlier analysis finished in {round( (time.time() - start_time)/60.0,3)} minutes...")

    @property
    def median_attribs(self):
        return self._median_attribs
    
    @property
    def uncertainty_attribs(self):
        return self._uncertainty_attribs
    
    @property
    def outlier_df(self):
        return self._outlier_df

    def _plot_outliers(self, outlier_val: float):
        self._input_data[self._outcome].plot(ylabel = " ".join(self._plot_save_name.split("_")), figsize=(15, 5))
        plt.vlines(np.arange(0, self._input_data.shape[0])[self._input_data[self._outcome] > outlier_val], \
                                                        self._input_data[self._outcome].min(), \
                                                        self._input_data[self._outcome].max(), linewidth=10, alpha=0.3, color='r')
        plt.xticks(np.arange(0, self._input_data.shape[0]), labels = [d.date() for d in self._input_data[self._date_column]], rotation=90, fontsize=6)
        if self._location_name is not None:
            plt.title(self._location_name.replace("_", " "))
        if "use_spark" not in list(self._config.keys()) or not self._config["use_spark"]:
            plt.savefig(f"{self._output_path}/{self._plot_save_name}.jpg", format = "JPG", bbox_inches='tight')            
        plt.close()

    def _process_outlier_df(self, median_attribs, uncertainty_attribs):
        outlier_df = []
        for k, v in median_attribs.items():
            outlier_df.append({"FEATURE": k,
                               "FEATURE_NAME": self._feature_to_name_map[k],
                               "MEDIAN": v,
                               "CONFIDENCE_INTERVAL_LB": uncertainty_attribs[k][0],
                               "CONFIDENCE_INTERVAL_UB": uncertainty_attribs[k][1]})
        return pd.DataFrame(outlier_df)

    def _run(self):
        outlier_val = np.percentile(self._input_data[self._outcome], self._outlier_percentile)
        print(f"Starting outlier analysis using outlier value of {round(outlier_val, 3)} ...")
        data_lower_val = self._input_data.sort_values(by=self._date_column,
                                                      ascending=True)[self._input_data[self._outcome] <= outlier_val][self._variables]
        data_higher_val = self._input_data.sort_values(by=self._date_column,
                                                      ascending=True)[self._input_data[self._outcome] > outlier_val][self._variables]
        self._plot_outliers(outlier_val)
        
    
        median_attribs, uncertainty_attribs = gcm.confidence_intervals(gcm.fit_and_compute(gcm.attribute_anomalies,
                                                                                            self._scm,
                                                                                            data_lower_val,
                                                                                            target_node=self._outcome,
                                                                                            anomaly_samples=data_higher_val),
                                                                                            num_bootstrap_resamples=self._number_bootstrap_resamples,
                                                                                            n_jobs = self._n_jobs)
        outlier_df = self._process_outlier_df(median_attribs, uncertainty_attribs)

        return median_attribs, uncertainty_attribs, outlier_df