
import os.path as ops
import argparse
import os
import sys
print(sys.path)
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

from attentive_gan_model import derain_drop_net
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--weights_path', type=str, default='ckpt/derain_gan.ckpt-100000')
    parser.add_argument('--label_path', type=str, default=None, help='The label image path')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention it suggests
    :param attention_map:
    :return:
    """
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8) + 255
    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map

    return attention_map_color


def test_model(image_path, weights_path, label_path=None):
    """

    :param image_path:
    :param weights_path:
    :param label_path:
    :return:
    """


    file_length = len(os.listdir(image_path))

    height,width,_ = 800,1200,3


    assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, height,width, 3],
                                  name='input_tensor'
                                  )
    import glob

    phase = tf.constant('test', tf.string)

    net = derain_drop_net.DeRainNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()
    files = glob.glob(image_path+'/*.png')
    masks = np.zeros((height,width), dtype=np.uint8)

    with sess.as_default():
        for image_path in files:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image_vis = image
            image = np.divide(np.array(image, np.float32), 127.5) - 1.0
            saver.restore(sess=sess, save_path=weights_path)

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for i in range(output_image.shape[2]):
                output_image[:, :, i] = minmax_scale(output_image[:, :, i])

            output_image = np.array(output_image, np.uint8)
            atte_maps_avg = (atte_maps[0][0, :, :, 0]+atte_maps[1][0, :, :, 0]+atte_maps[2][0, :, :, 0]+atte_maps[3][0, :, :, 0])/4
            masks=masks+atte_maps_avg
        
        masks=masks/file_length
        save_path = str(Path(args.image_path).parent)+ '/vis'
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path+'/masks.npy', masks)
        plt.figure('atte_map_avg')
        plt.imshow(masks,cmap='jet')
        plt.savefig(save_path+'/mask_heat.png')
        plt.show()
        masks = np.where((masks >= 0.5) & (masks <= 0.7), 255, 0)

        cv2.imwrite(save_path+'/masks_cv.png',masks)

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test model
    test_model(args.image_path, args.weights_path, args.label_path)
