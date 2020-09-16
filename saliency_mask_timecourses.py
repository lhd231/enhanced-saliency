import numpy as np
import sys

def relu(x):
    x[x<0] = 0
    return x


saliency_path = sys.argv[1]
input_data_path = sys.argv[2]
output_path = sys.argv[3]

saliency = np.load(saliency_path)
data = np.load(input_data_path)

masked_data = []
print(saliency.shape)
print(data.shape)
for i in range(saliency.shape[0]):
    sal_sub = saliency[i]
    sal_sub = relu(sal_sub)
    sal_sub = sal_sub/np.max(sal_sub)

    data_sub = data[i]
    data_sub = data_sub * sal_sub
    masked_data.append(data_sub)

masked_data = np.array(masked_data)

np.save(output_path, masked_data)
