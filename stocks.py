import tensorflow as tf

import collections
import numpy as np
import pandas as pd

defaults = collections.OrderedDict([
  ("date", [""]),
  ("open", ["0"]),
  ("high", ["0"]),
  ("low", ["0"]),
  ("close", ["0"]),
  ("adj-close", ["0"]),
  ("volume", ["0"])
])

types = collections.OrderedDict((key, type(value[0])) for key, value in defaults.items())

def decode_line(line):
  """Convert a csv line into a (features_dict,label) pair."""
  # Decode the line to a tuple of items based on the types of
  # csv_header.values().
  items = tf.decode_csv(line, list(defaults.values()))

  # Convert the keys and items to a dict.
  pairs = zip(defaults.keys(), items)
  features_dict = dict(pairs)

  # Remove the label from the features_dict
  label = features_dict.pop("high")

  return features_dict, label

def in_training_set(line):
  """Returns a boolean tensor, true if the line is in the training set."""
  # If you randomly split the dataset you won't get the same split in both
  # sessions if you stop and restart training later. Also a simple
  # random split won't work with a dataset that's too big to `.cache()` as
  # we are doing here.
  num_buckets = 1000000
  train_fraction = 0.7
  bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
  # Use the hash bucket id as a random number that's deterministic per example
  return bucket_id < int(train_fraction * num_buckets)

def in_test_set(line):
  """Returns a boolean tensor, true if the line is in the training set."""
  # Items not in the training set are in the test set.
  # This line must use `~` instead of `not` beacuse `not` only works on python
  # booleans but we are dealing with symbolic tensors.
  return ~in_training_set(line)

def dataset():
  base_dataset = (tf.data
                  # Get the lines from the file.
                  .TextLineDataset("GOOG.csv"))

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Cache data so you only read the file once.
           .cache()
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line))

  # Do the same for the test-set.
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test

def raw_dataframe():
  """Load the data as a pd.DataFrame"""
  return pd.read_csv("GOOG.csv", names=types.keys(), dtype=types)

def load_data(y_name="high", train_fraction=0.7, seed=None):
  """
  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_imports85_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """

  # Load the raw data columns
  data = raw_dataframe()

  # Shuffle the data
  # np.random.seed(seed)

  # Split the data into train/test subsets
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)

def main():
  train, test = load_data()
  print("train", train)
  print("test", test)

if __name__ == "__main__":
  main()

# filename_queue = tf.train.string_input_producer(["GOOG.csv"])
# reader = tf.TextLineReader(skip_header_lines=1)
# key, value = reader.read(filename_queue)

# # Default values, in case of empty columns. Also specifies the type of the decoded result
# record_defaults = [
#   tf.constant([], dtype=tf.string),
#   tf.constant([], dtype=tf.float32),
#   tf.constant([], dtype=tf.float32),
#   tf.constant([], dtype=tf.float32),
#   tf.constant([], dtype=tf.float32),
#   tf.constant([], dtype=tf.float32),
#   tf.constant([], dtype=tf.float32)
# ]
# col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.stack([col2, col3, col4, col5])

# with tf.Session() as sess:
#   # Start populating the filename queue.
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)

#   for i in range(1200):
#     # Retrieve a single instance:
#     example, label = sess.run([features])
#     print(example, label)

#   coord.request_stop()
#   coord.join(threads)