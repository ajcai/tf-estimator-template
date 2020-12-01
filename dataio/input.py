import tensorflow as tf

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10 

def parse_record(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
    image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
    
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label

def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_with_crop_or_pad(
            image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)
        
        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.image.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

def generate_input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames=file_names)
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Transformation
        dataset = dataset.map(parse_record)
        dataset = dataset.map(
            lambda image, label: (preprocess_image(image, is_training), label))
        
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2 * batch_size)
        
        dataset = dataset.map(
            lambda images, labels:({'images': images}, labels)
        )
        return dataset
    
    return _input_fn

def get_feature_columns():
    feature_columns = {
        'images': tf.feature_column.numeric_column('images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
    }
    return feature_columns

def serving_input_fn():
    receiver_tensor = {'images': tf.compat.v1.placeholder(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32)}
    features = {'images': tf.map_fn(preprocess_image, receiver_tensor['images'])}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
