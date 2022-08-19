import warnings
import dataInit
import originalModel
import improvedModel
import torch
import time

warnings.filterwarnings("ignore")
labels, feature_list, input_features = dataInit.data_init(is_draw=False)

start = time.perf_counter()

# originalModel.original_model(labels, feature_list, input_features)
improvedModel.improved_model(labels, feature_list, input_features, input_size=input_features.shape[1],
                             hidden_size=128, output_size=1, batch_size=16)

end = time.perf_counter()
print(str(end-start)+' s')
