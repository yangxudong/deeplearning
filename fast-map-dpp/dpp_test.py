from dpp import *

import time

item_size = 5000
feature_dimension = 5000
max_length = 1000

scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)
feature_vectors = np.random.randn(item_size, feature_dimension)

feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
similarities = np.dot(feature_vectors, feature_vectors.T)
kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))

print 'kernel matrix generated!'

t = time.time()
result = dpp(kernel_matrix, max_length)
print 'algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t)
