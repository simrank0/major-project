from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_slim as slim
import os
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time
import numpy as np

Flags = tf.app.flags

#Want graphs built. DON'T calculate values of tensors as they occur
#FLAG -> causing tensors to act before they 
tf.disable_eager_execution()
# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                                                 'be 0. If set False, you are going to continue the training. That is, '
                                                 'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('pre_trained_model_type', 'The type of pretrained model')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
Flags.DEFINE_string('task', None, 'The task: SRGAN, SRResnet')
# The data preparing operation
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the high resolution input data')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# The testing mode
if FLAGS.mode == 'test':
    # TESTING
    exit()


# the inference mode (just perform super resolution on the input image)
elif FLAGS.mode == 'inference':
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None

    # Declare the test data reader
    inference_data = inference_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        max_iter = len(inference_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
            path_lr = inference_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
            filesets = save_images(results, FLAGS)
            for i, f in enumerate(filesets):
                print('evaluate image', f['name'])


# The training mode
elif FLAGS.mode == 'train':
    # Load data for training and testing
    # ToDo Add online downscaling
    data = data_loader(FLAGS)
    print('Data count = %d' % (data.image_count))

    # Connect to the network
    if FLAGS.task == 'SRGAN':
        Net = SRGAN(data.inputs, data.targets, FLAGS)
    else:
        raise NotImplementedError('Unknown task type')

    print('Finish building the network!!!')

    # Convert the images output from the network
    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(data.inputs)
        targets = deprocess(data.targets)
        outputs = deprocess(Net.gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    # Compute PSNR
    

    # Add image summaries
    

    # Add scalar summary


    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=10)

    # The variable list
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    print('Optimization starts!!!')
    start = time.time()
    for step in range(max_iter):
        fetches = {
            "train": Net.train,
            "global_step": sv.global_step,
        }

        if ((step + 1) % FLAGS.summary_freq) == 0:
            print('Recording summary!!')
            sv.summary_writer.add_summary(results['summary'], results['global_step'])


        if ((step +1) % FLAGS.save_freq) == 0:
            print('Save the checkpoint')
            saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')