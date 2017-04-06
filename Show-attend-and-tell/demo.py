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
from PIL import Image
from core.utils import decode_captions
from resize import resize_image

plt.rcParams['figure.figsize'] = (20.0, 15.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

def preprocess(image):
    image = resize_image(image)
    return image

#Load and build VGG network
vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()
vggnet_variables = set(tf.all_variables())

#Build CaptionGenerator
model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                        dim_hidden=1024, n_time_step=16, prev2out=True,
                        ctx2out=True, alpha_c=1.0, selector=True, dropout=True, vggnet=vggnet)

alphas, betas, sampled_captions = model.build_sampler(max_len=20)

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

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Aproximately 2GB og GPU memory is required to run the VGG+SAT network
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess  = tf.Session(config=config)

saver = tf.train.Saver(set(tf.all_variables())-vggnet_variables) #Restore only caption-generator variables
restore_path = './model/lstm/model-20'
saver.restore(sess, restore_path)


with sess.as_default():
    tf.initialize_variables(vggnet_variables).run() #only initialize the newly added VGGnet variables
    print([var.name for var in tf.all_variables()])

    for idx in range(1,9):
        t0 = time.time()

        #receive image
        #TODO: Receive from client
        image = Image.open('./samples/%d.jpg'%idx)
        t1 = time.time()
        print('Image Read time: ', t1- t0)

        #preprocess image
        image = np.array(preprocess(image), dtype=np.float32)
        t2 = time.time()
        print('Image pre-process time: ', t2- t1)

        #Get caption
        alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict={vggnet.images: [image]})
        decoded = decode_captions(sam_cap, model.idx_to_word)
        t4 = time.time()
        print('Image captioning time: ', t4- t2)
        print("Caption %d: %s"%(idx,decoded))

        print('total time: ', time.time()-t0)

        #send caption
        #TODO: Send caption to client

sess.close()
