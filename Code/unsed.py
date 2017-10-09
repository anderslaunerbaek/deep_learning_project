

def tf_conv2d(inputs, name, 
              filters=64, kernel_size=(3,3), 
              padding='same', activation=tf.nn.relu, 
              trainable=TRAIN_FEATURES):
    return(tf.layers.conv2d(inputs=inputs,
                            bias_initializer=tf.constant_initializer(weights_dict[name + '_b']),
                            kernel_initializer=tf.constant_initializer(weights_dict[name + '_W']),
                            name=name,
                            filters=filters, 
                            kernel_size=kernel_size,
                            padding=padding, 
                            activation=activation,
                            trainable=trainable))

def tf_max_pooling2d(inputs, name,
                     pool_size=(2,2),
                     strides=(2,2)):
    return(tf.layers.max_pooling2d(inputs=inputs,
                                     name=name,
                                     pool_size=pool_size,
                                     strides=strides))

def tf_max_pooling2d(inputs, name,
                     pool_size=(2,2),
                     strides=(1,1)):
    return(tf.layers.max_pooling2d(inputs=inputs, 
                                     name=name,
                                     pool_size=pool_size,
                                     strides=strides))
# init model
tf.reset_default_graph()

# init placeholders
x_pl = tf.placeholder(tf.float32, [None, HEIGTH, WIDTH, NCHANNELS], name='input_placeholder')
y_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='target_placeholder')
#y_pl = tf.cast(y_pl, tf.float32)

with tf.variable_scope('VVG16_layer'):
    # level one
    conv1_1 = tf_conv2d(inputs=x_pl, name='conv1_1')
    conv1_2 = tf_conv2d(inputs=conv1_1, name='conv1_2')
    pool1 = tf_max_pooling2d(inputs=conv1_2, name='pool1')
    
    # level two
    conv2_1 = tf_conv2d(inputs=pool_1, name='conv2_1', filters=128)
    conv2_2 = tf_conv2d(inputs=conv2_1, name='conv2_2', filters=128)
    pool2 = tf_max_pooling2d(inputs=conv2_2, name='pool_2')
    
    # level three
    conv3_1 = tf_conv2d(inputs=pool_2, name='conv3_1', filters=256)
    conv3_2 = tf_conv2d(inputs=conv3_1, name='conv3_2', filters=256)
    conv3_3 = tf_conv2d(inputs=conv3_2, name='conv3_3', filters=256)
    pool3 = tf_max_pooling2d(inputs=conv3_3, name='pool_3')
    
    # level four
    conv4_1 = tf_conv2d(inputs=pool_3, name='conv4_1', filters=512)
    conv4_2 = tf_conv2d(inputs=conv4_1, name='conv4_2', filters=512)
    conv4_3 = tf_conv2d(inputs=conv4_2, name='conv4_3', filters=512)
    pool4 = tf_max_pooling2d(inputs=conv4_3, name='pool_4')
    
    # level five
    conv5_1 = tf_conv2d(inputs=pool_4, name='conv5_1', filters=512)
    conv5_2 = tf_conv2d(inputs=conv5_1, name='conv5_2', filters=512)
    conv5_3 = tf_conv2d(inputs=conv5_2, name='conv5_3', filters=512)
    pool5 = tf_max_pooling2d(inputs=conv5_3, name='pool_5')
    
    # level six
    fc6 = tf.layers.dense(inputs=pool_5, name='fc6', units=4096)
    fc6_dropout = tf.layers.dropout(inputs=fc6, name='fc6_dropout',rate=0.5)
    
    # level seven
    fc7 = tf.layers.dense(inputs=fc6_dropout, name='fc7', units=4096)
    fc7_dropout = tf.layers.dropout(inputs=fc7, name='fc7_dropout',rate=0.5)
    
    # level eigth
    fc8 = tf.layers.dense(inputs=fc7_dropout, name='fc8', units=NUM_CLASSES)

#with tf.variable_scope('RRN_layer'):


with tf.variable_scope('output_layer'):    
    l_out = tf.nn.softmax(fc8, name='l_out')

print('Model consits of ', utils_DL.num_params(), 'trainable parameters.')