import pyspark
import sys
import numpy as np
import math
from datetime import datetime
import scipy.sparse as sps
import string
from collections import defaultdict

step_size = 2e-6
reg_param = 0.01

start_time = datetime.now()

def log(format_string, *args):
    curr_time = datetime.now()
    seconds_elapsed = (curr_time - start_time).seconds
    formatter = string.Formatter()
    header = "[%ds] " % seconds_elapsed
    full_format_string = header + format_string
    print full_format_string % args

def parse_line(line):
    parts = line.split()
    label = int(parts[0])
    label = 0 if label == -1 else label
    idx = 1
    feature_ids = []
    feature_vals = []
    for part in parts[1:]:
	feature = part.split(":")
        feature_ids.append(int(feature[0]) -  1)
        feature_vals.append(float(feature[1]))
    return (label, (np.array(feature_ids), np.array(feature_vals)))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def safe_log(x):
    if x < 1e-15:
	x = 1e-15
    return math.log(x)

def gd_partition(samples):
    local_updates = defaultdict(float)
    local_weights_array = weights_array_bc.value
    cross_entropy_loss = 0
    num_samples_processed = 0
    for sample in samples:
	label = sample[0]
	features = sample[1]
        feature_ids = features[0]
        feature_vals = features[1]
        local_weights = np.take(weights_array, feature_ids)                
	pred = sigmoid(feature_vals.dot(local_weights))
	diff = label - pred
	gradient = diff * feature_vals + reg_param * local_weights
        sample_update = step_size * gradient
        for i in range(0, _feature_ids.size):
            local_updates[feature_ids[i]] += sample_update[i]

	if label == 1:
	    cross_entropy_loss -= safe_log(pred)
	else:
	    cross_entropy_loss -= safe_log(1 - pred)
        num_samples_processed += 1
        if num_samples_processed % 10000 == 0:
            print "number of samples processed = %d" % num_samples_processed
    accumulated_updates = sps.csr_matrix(\
                                         (local_updates.values(), local_updates.keys(), [0, len(local_updates)]), \
                                         shape=(1, num_features))
    return [(cross_entropy_loss, accumulated_updates)]

if __name__ == "__main__":
    data_path = sys.argv[1]
    num_features = int(sys.argv[2])
    num_cores = 32
    num_partitions = num_cores*16
    conf = pyspark.SparkConf().setAppName("SparseLogisticRegressionGD")
    sc = pyspark.SparkContext(conf=conf)
    text_rdd = sc.textFile(data_path, minPartitions=num_partitions)

    samples_rdd = text_rdd.map(parse_line, preservesPartitioning=True)\
                 .persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK)
    num_samples = samples_rdd.count()
    print "number of samples = %d" % num_samples
    weights_array = np.ones(num_features) * 0.001
    log("created weight locally")
    loss_list = []
    for iteration in range(0, 2):
        weights_array_bc = sc.broadcast(weights_array)
        loss_updates_rdd = samples_rdd.mapPartitions(gd_partition)
        ret = loss_updates_rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        loss = ret[0]
        updates = ret[1]
        loss_list.append(loss)
        weights_array_bc.destroy()
        weights_array += updates.toarray().squeeze()
        step_size *= 0.95
	log("iteration: %d, cross-entropy loss: %f" % (iteration, loss))
    print loss_list
