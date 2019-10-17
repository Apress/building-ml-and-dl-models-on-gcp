import six

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

# Define the format of your input data including unused columns.
CSV_COLUMNS = [
    'sepal_length', 'sepal_width', 'petal_length',
    'petal_width', 'class'
]
CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], ['']]
LABEL_COLUMN = 'class'
LABELS = ['0', '1', '2']

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Continuous base columns.
    tf.feature_column.numeric_column('sepal_length'),
    tf.feature_column.numeric_column('sepal_width'),
    tf.feature_column.numeric_column('petal_length'),
    tf.feature_column.numeric_column('petal_width')
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
    {LABEL_COLUMN}

def generate_model_fn(hidden_units=None, learning_rate=None, params={}):
    """Generates a model_fn for a feed forward classification network.
    Takes hyperparameters that define the model and returns a model_fn that
    generates a spec from input Tensors.
    Args:
        hidden_units (list): Hidden units of the DNN.
        learning_rate (float): Learning rate for the SGD.
    Returns:
        A model_fn.
    """
    
    def _model_fn(mode, features, labels):
        """A model_fn that builds the DNN classification spec.
        Args:
        mode (tf.estimator.ModeKeys): One of ModeKeys.(TRAIN|PREDICT|INFER) which
            is used to selectively add operations to the graph.
        features (Mapping[str:Tensor]): Input features for the model.
        labels (Tensor): Label Tensor.
        Returns:
        tf.estimator.EstimatorSpec which defines the model. Will have different
        populated members depending on `mode`.
        """
        (sepal_length, sepal_width, petal_length, petal_width) = INPUT_COLUMNS
        
        transformed_columns = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]

        inputs = tf.feature_column.input_layer(features, transformed_columns)
        label_values = tf.constant(LABELS)

        # Build the DNN.
        curr_layer = inputs

        # hidden_units=[
        #     max(2, int(params.get('first_layer_size') * params.get('scale_factor')**i))
        #     for i in range(params.get('num_layers'))
        # ]

        for layer_size in hidden_units:
            curr_layer = tf.layers.dense(
                curr_layer,
                layer_size,
                activation=tf.nn.relu,
                # This initializer prevents variance from exploding or vanishing when
                # compounded through different sized layers.
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            )

        # Add the output layer.
        logits = tf.layers.dense(
            curr_layer,
            len(LABELS),
            # Do not use ReLU on last layer
            activation=None,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        if mode in (Modes.PREDICT, Modes.EVAL):
            probabilities = tf.nn.softmax(logits)
            predicted_indices = tf.argmax(probabilities, 1)

        batch_size = params.get('batch_size', 64)

        if mode in (Modes.TRAIN, Modes.EVAL):
            # Convert the string label column to indices.
            # Build a lookup table inside the graph.
            table = tf.contrib.lookup.index_table_from_tensor(label_values)

            # Use the lookup table to convert string labels to ints.
            label_indices = table.lookup(labels)
            # Make labels a vector
            label_indices_vector = tf.squeeze(label_indices, axis=[1])

            # global_step is necessary in eval to correctly load the step
            # of the checkpoint we are evaluating.
            global_step = tf.contrib.framework.get_or_create_global_step()
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=label_indices_vector))
            # tf.summary.scalar('loss', loss)

        if mode == Modes.PREDICT:
            # Convert predicted_indices back into strings.
            predictions = {
                'classes': tf.gather(label_values, predicted_indices),
                'scores': tf.reduce_max(probabilities, axis=1)
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs)

        if mode == Modes.TRAIN:
            # Build training operation.
            train_op = tf.contrib.tpu.CrossShardOptimizer(tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=3.0,
                l2_regularization_strength=10.0)).minimize(
                    loss, global_step=global_step)
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

        if mode == Modes.EVAL:
            # Return accuracy and area under ROC curve metrics
            labels_one_hot = tf.one_hot(
                label_indices_vector,
                depth=label_values.shape[0],
                on_value=True,
                off_value=False,
                dtype=tf.bool)
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
                'auroc': tf.metrics.auc(labels_one_hot, probabilities)
            }
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return _model_fn

# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************

def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    features = _decode_csv(csv_row)
    # Ignore label column
    features.pop(LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features,
                                              {'csv_row': csv_row})

def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
    )
    features = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS))
    return tf.estimator.export.ServingInputReceiver(
      features, {'example_proto': example_bytestring})


# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# [END serving-function]

SERVING_FUNCTIONS = {
  'JSON': json_serving_input_fn,
  'EXAMPLE': example_serving_input_fn,
  'CSV': csv_serving_input_fn
}

def _decode_csv(line):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""

    # Takes a rank-1 tensor and converts it into rank-2 tensor
    # row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))

    # Remove unused columns
    for col in UNUSED_COLUMNS:
        features.pop(col)
    for key, value in six.iteritems(features):
        features[key] = tf.expand_dims(features[key], -1)
    return features

def input_fn(params,
         filenames,
         num_epochs=None,
         shuffle=True,
         skip_header_lines=1):
    """Generates features and labels for training or evaluation.
    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.
    Args:
      filenames: [str] A List of CSV file(s) to read data from.
      num_epochs: (int) how many times through to read the data. If None will
        loop through data indefinitely
      shuffle: (bool) whether or not to randomize the order of data. Controls
        randomization of both file order and line order within files.
      skip_header_lines: (int) set to non-zero in order to skip header lines in
        CSV files.
      batch_size: (int) First dimension size of the Tensors returned by input_fn
    Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
    """

    batch_size = params.get('batch_size', 64)

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
      _decode_csv)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    # iterator = dataset.repeat(num_epochs).batch(
    #     batch_size).make_one_shot_iterator()
    # features = iterator.get_next()
    # return features, features.pop(LABEL_COLUMN)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset