import tensorflow as tf

def inference(images):
    # 1st Convolutional Layer
    conv1 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=[5, 5], padding='same', 
        activation=tf.nn.relu, name='conv1')(images)
    pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=[3,3], strides=2, name='pool1')(conv1)
    norm1 = tf.nn.lrn(
        pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    # 2nd Convolutional Layer
    conv2 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=[5, 5], padding='same', 
        activation=tf.nn.relu, name='conv2')(norm1)
    norm2 = tf.nn.lrn(
        conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=[3,3], strides=2, name='pool2')(norm2)
    
    # Flatten Layer
    shape = pool2.get_shape()
    pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])
    
    # 1st Fully Connected Layer
    dense1 = tf.keras.layers.Dense(
        units=384, activation=tf.nn.relu, name='dense1')(pool2_)
    
    # 2nd Fully Connected Layer
    dense2 = tf.keras.layers.Dense(
        units=192, activation=tf.nn.relu, name='dense2')(dense1)
    
    # 3rd Fully Connected Layer (Logits)
    logits = tf.keras.layers.Dense(
        units=10, activation=tf.nn.relu, name='logits')(dense2)
    
    return logits
    