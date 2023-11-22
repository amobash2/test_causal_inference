from causica.sem.sem_distribution import SEMDistributionModule
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Dict
import time

class CausalGraph:
    def __init__(self,
                 logger,
                 config,
                 sem_module: SEMDistributionModule,
                 data_module,
                 feature_to_name_map: Dict,
                 output_path: str):
        self._logger = logger
        self._config = config
        self._feature_to_name_map = feature_to_name_map
        self._output_path = output_path
        sem = sem_module().mode
        start_time = time.time()
        graph = nx.from_numpy_array(sem.graph.cpu().numpy(), create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, dict(enumerate(data_module.dataset_train.keys())))

        labels = {node: i for i, node in enumerate(graph.nodes)}
        layout = nx.layout.spring_layout(graph)

        _, axis = plt.subplots(1, 1, figsize=(8, 8))
        plt.rcParams.update({'font.size': 8})

        if self._feature_to_name_map is None:
            if self._logger: self._logger.error("Feature to name map is empty...")
            raise Exception("Feature to name map is empty...")

        for node, i in labels.items():
            try:
                axis.scatter(layout[node][0], layout[node][1], label=f"{i}: {self._feature_to_name_map[node]}")
            except:
                axis.scatter(layout[node][0], layout[node][1], label=f"{i}: {node}")
        axis.legend()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        nx.draw_networkx(graph, pos=layout, with_labels=True, arrows=True, labels=labels, ax = axis)
        plt.savefig(f"{self._output_path}/causal_graph.jpg", format = "JPG", bbox_inches='tight')
        plt.close()
        self._graph = graph
        if self._logger: self._logger.info(f"Causal graph processing finished in {round( (time.time() - start_time)/60.0,3)} minutes...")

    @property
    def graph(self):
        return self._graph

    def find_cycles(self):
        cycles = []
        for c in sorted(nx.simple_cycles(self._graph)):
            cyl = ""
            for idx, d in enumerate(c):
                try:
                    cyl += self._feature_to_name_map[d] + f" ({d})"
                except:
                    cyl += d + f" ({d})"
                if idx < len(c) - 1:
                    cyl += " --> "
            cycles.append(cyl)
        return cycles
    
    def store_graph(self):
        df = nx.to_pandas_edgelist(self._graph)

        updated_df = []
        for _, row in df.iterrows():
            updated_df.append({"SOURCE": row["source"],
                               "SOURCE_NAME": self._feature_to_name_map[row["source"]] if row["source"] in list(self._feature_to_name_map.keys()) else row["source"],
                               "TARGET": row["target"],
                               "TARGET_NAME": self._feature_to_name_map[row["target"]] if row["target"] in list(self._feature_to_name_map.keys()) else row["target"],
                               "WEIGHT": 1})
        updated_df = pd.DataFrame(updated_df)
        updated_df.to_csv(f"{self._output_path}/network_edgelist.csv", index=None)
        print(f"Graph dataframe is stored at {self._output_path}")
        
        return updated_df
