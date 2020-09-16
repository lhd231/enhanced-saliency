import numpy as np
import sys

data_path = sys.argv[1]
window_length = int(sys.argv[2])
output_path = sys.argv[3]

N = np.load(data_path)

M = np.zeros([N.shape[0],N.shape[1]-window_length,1431])

for i,s in enumerate(N):
    empty = np.zeros([s.shape[0]-window_length,1128])
    for t in range(s.shape[0]-window_length):
        empty[t] = np.corrcoef(s[t:t+window_length].T)[np.tril_indices(53)]

    M[i,:,:] = empty
print(np.sum(M==0))
np.save(output_path,M)

