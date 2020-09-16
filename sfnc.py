import numpy as np
import sys

data_path = sys.argv[1]
output_path = sys.argv[2]

data = np.load(data_path)

fncs = []
print(data.shape)
for i in range(data.shape[0]):
    x = data[i]
    x = np.corrcoef(x.T)[np.tril_indices(47)]
    x[np.isnan(x)] = 0
    print(np.sum(x))
    fncs.append(x)
    #print(x.shape)

fncs = np.array(fncs)

fncs[np.isnan(fncs)] = 0

np.save(output_path, fncs)