import numpy as np

a = [1,2,3,4,5]
q = [i / 8 for i in range(8)]
quantile_bucket = np.quantile(a, q, axis=None)

min_value, max_value = min(a) - (1e-5), max(a) + (1e-5)
quantile_bucket = [min_value] + quantile_bucket + [max_value]

quant_update = a
for j in range(len(quantile_bucket) - 1):
    locations_bucket = (quant_update > quantile_bucket[j]) & (quant_update <= quantile_bucket[j + 1])
    quant_update[locations_bucket] = (quantile_bucket[j] + quantile_bucket[j + 1]) / 2

print(quant_update)