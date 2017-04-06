from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate
from core.vggnet import Vgg19
import time
from core.utils import decode_captions
from resize import resize_image
import skimage.transform
from Data import DataIn, DataOut

plt.rcParams['figure.figsize'] = (20.0, 15.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
dataIn  = DataIn('image')
dataOut = DataOut('console')

#Load and build VGG network
vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()
vggnet_variables = set(tf.all_variables())

#Build CaptionGenerator
model = CaptionGenerator(dim_feature=[196, 512], dim_embed=512,
                        dim_hidden=1024, n_time_step=16, prev2out=True,
                        ctx2out=True, alpha_c=1.0, selector=True, dropout=True, vggnet=vggnet)

alphas, betas, sampled_captions = model.build_sampler(max_len=20)

#Aproximately 2GB og GPU memory is required to run the VGG+SAT network
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess  = tf.Session(config=config)

saver = tf.train.Saver(set(tf.all_variables())-vggnet_variables) #Restore only caption-generator variables
restore_path = './model/lstm/model-20'
saver.restore(sess, restore_path)



def visualize(alps, bts, words, img):
    # Plot images with attention weights 
    plt.subplot(4, 5, 1) #max 4 x 5 = 20 words
    plt.imshow(img)
    plt.axis('off')

    words = decoded[0].split(" ")
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t+2)
        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[t]) , color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(img)
        alp_curr = alps[0,t,:].reshape(14,14)
        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
        plt.imshow(alp_img, alpha=0.85)
        plt.axis('off')
    plt.show()



with sess.as_default():
    tf.initialize_variables(vggnet_variables).run() #only initialize the newly added VGGnet variables
    print([var.name for var in tf.all_variables()])

    for raw_image in dataIn.get():
        image = np.array(raw_image, dtype=np.float32)
        t0 = time.time()
        #Get caption
        alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict={vggnet.images: [image]})
        decoded = decode_captions(sam_cap, model.idx_to_word)
        t1 = time.time()
        print('Image captioning time: ', t1- t0)

        visualize(alps, bts, decoded, raw_image)
        dataOut.send(decoded)

        print('total time: ', time.time()-t0)
    
sess.close()






"""
#Get caption for Batch (test/valid) of images.
data = load_coco_data(data_path='./data', split='test')
solver = CaptioningSolver(model, data, data, n_epochs=15, batch_size=128, update_rule='adam',
                                      learning_rate=0.0025, print_every=2000, save_every=1, image_path='./image/val2014_resized',
                                pretrained_model=None, model_path='./model/lstm', test_model='./model/lstm/model-20',
                                 print_bleu=False, log_path='./log/')

#solver.test(data, split='val')
#evaluate(data_path='./data', split='val')
solver.test(data, split='test')
evaluate(data_path='./data', split='test')
"""
