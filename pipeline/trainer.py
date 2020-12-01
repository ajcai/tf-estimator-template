import os
import sys
import shutil
sys.path.append('./')

from configs.config import *
from dataio.input import *
from models.model import *


def model_fn(features, labels, mode, params):
    # Create the input layers from the features
    feature_columns = list(get_feature_columns().values())
    
    images = tf.compat.v1.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    
    images = tf.reshape(
        images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    
    # Calculate logits through CNN
    logits = inference(images)
    
    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.compat.v1.train.get_or_create_global_step()
        label_indices = tf.argmax(input=labels, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', loss)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['lr'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.compat.v1.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def train():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    
    model_dir = 'trained_models/{}'.format(Config.model_name)
    train_data_files = ['datasets/cifar-10/train.tfrecords']
    valid_data_files = ['datasets/cifar-10/validation.tfrecords']

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=Config.log_interval_steps,
        save_summary_steps=Config.log_interval_steps,
        save_checkpoints_steps=Config.save_checkpoints_steps,
        tf_random_seed=Config.tf_random_seed,
        model_dir=model_dir,
        session_config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,
                                                force_gpu_compatible=True,
                                                visible_device_list='0'
                                               )
        ),
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn, 
                                       config=run_config, 
                                       params={'lr': Config.learning_rate})

    # There is another Exporter named FinalExporter
    exporter = tf.estimator.LatestExporter(
        name='Servo',
        serving_input_receiver_fn=serving_input_fn,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5)

    train_spec = tf.estimator.TrainSpec(
        input_fn=generate_input_fn(file_names=train_data_files,
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  batch_size=Config.batch_size),
        max_steps=Config.max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=generate_input_fn(file_names=valid_data_files,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=Config.batch_size),
        steps=Config.eval_steps, exporters=exporter) 

    if not Config.use_checkpoint:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)


    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def evaluate():
    model_dir = 'trained_models/{}'.format(Config.model_name)
    test_data_files = ['datasets/cifar-10/eval.tfrecords']
    
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=Config.log_interval_steps,
        save_summary_steps=Config.log_interval_steps,
        save_checkpoints_steps=Config.save_checkpoints_steps,
        tf_random_seed=Config.tf_random_seed,
        model_dir=model_dir,
        session_config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,
                                                force_gpu_compatible=True,
                                                visible_device_list='0'
                                               )
        ),
    )
    
    test_input_fn = generate_input_fn(file_names=test_data_files,
                                 mode=tf.estimator.ModeKeys.EVAL,
                                 batch_size=1000)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    print(estimator.evaluate(input_fn=test_input_fn, steps=1))
    
def predict():
    model_dir = 'trained_models/{}'.format(Config.model_name)
    export_dir = model_dir + '/export/Servo/'
    saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1]) 
    
    import _pickle as cPickle
    import numpy
    data_dict = cPickle.load(open('datasets/cifar-10/cifar-10-batches-py/test_batch', 'rb'), 
                             encoding='iso-8859-1')
    N = 1000
    images = data_dict['data'][:N].reshape([N, 3, 32, 32]).transpose([0, 2, 3, 1])
    labels = data_dict['labels'][:N]
    
    predictor_fn = tf.saved_model.load(saved_model_dir).signatures["predictions"]
    # output = predictor_fn({'images': tf.convert_to_tensor(images, dtype=tf.float32)})
    output = predictor_fn(tf.convert_to_tensor(images, dtype=tf.float32))
    predictions = output['classes'].numpy()
    accuracy = numpy.sum(
      [ans==ret for ans, ret in zip(labels, predictions)]) / float(N)

    print('test accuracy:', accuracy)
    