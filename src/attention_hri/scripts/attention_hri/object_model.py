from __future__ import print_function
from __future__ import absolute_import

# attempt to automatically switch backend
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D, Conv2D, TimeDistributed, UpSampling2D
except:
    import tflite_runtime.interpreter as tflite

# from keras.utils.data_utils import get_file
import tensorflow as tf
import numpy as np

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5,
          1e-5, 1e-6, 1e-6, 1e-7, 1e-7]
    return lr[epoch]

def mods_xy(use_tflite=False):
    if use_tflite == False:
        model = Sequential()
        # conv_1
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224, 224, 3)))#batch__input_shape=(224,224,3)
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(256, 320, 3)))#batch__input_shape=(224,224,3)

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  #180X128*64

        # conv_2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')) #90X64X128

        # conv_3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')) # 32x40X256

        # conv_4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        #
        model.add(MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same')) #32X40X512

        # conv_5
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', dilation_rate=(2, 2)))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', dilation_rate=(2, 2)))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', dilation_rate=(2, 2)))
        #
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

        # Load weights
        weights_path = '/home/maelic/Documents/NATNAEL/catkin_ws/src/attention_hri/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        
        model.load_weights(weights_path)
        return model
    else:
        # Load TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path="/home/maelic/Documents/NATNAEL/catkin_ws/src/attention_hri/resource/VGG16_model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter #, input_details, output_details
    
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

    # # Save the model.
    # with open('/home/maelic/Documents/NATNAEL/catkin_ws/src/attention_hri/resource/VGG16_model.tflite', 'wb') as f:
    #   f.write(tflite_model)

    # print("Model Summary")
    # print(model.summary())
    # return model

# mods_xy().summary()

def transform_saliency(data, stateful, use_tflite=False):
    xyshift_vgg = mods_xy(use_tflite=use_tflite)

    outs = TimeDistributed(xyshift_vgg)(data[2]) # 1 is weighted overlay of xy and raw data
    print("Inside Model 1 ", outs.shape)

    # outs2 = TimeDistributed(xyshift_vgg)(data[2]) # for inter frame differencing
    # outs3 = TimeDistributed(xyshift_vgg)(data[2])

    # print(outs2.shape, "Shape after first vgg-16")

    # outs = Add()([outs1, outs2]) # first residual link between raw and xy-shift

    print(outs.shape, "Shape after first residual with raw and xyshift")

    # outs = Add()([outs, outs3])
    print(outs.shape, "Shape after second residual with three frame")

    # outs = Add()([outs, data[2]])
    # outs2 = Add()([data[0], data[2]])
    #
    # outs = Add()([outs1, outs2])

    # outs = TimeDistributed(xyshift_vgg)(outs)


    print(outs.shape, "Shape after first data[1] vgg-16")
    # print(outs3.shape, "Shape after first data[2] vgg-16")

    # outs2 = TimeDistributed(xyshift_vgg)(data[1])
    # outs = outs + outs2
    # print(outs.shape, "changed shape")


    # concatination approach
    # residual approach
    #

    outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(outs)

    print("Shape after first convolution", outs.shape)
    attention = TimeDistributed(UpSampling2D(2))(outs)
    print(" first upsample", attention.shape)

    outs = TimeDistributed(UpSampling2D(4))(outs)
    print("second upsample", outs.shape)

    print(outs.shape, attention.shape)


    return [outs]