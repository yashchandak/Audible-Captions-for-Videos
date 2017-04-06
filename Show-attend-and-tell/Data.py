from __future__ import print_function
import numpy as np
import time
from PIL import Image
from resize import resize_image

class DataIn:
    def __init__(self,source):
        if source == 'video':
            self.fn = self.getVideo
        elif source == 'image':
            self.fn = self.getImage
        elif source == 'client':
            self.fn = self.getClient
        elif source == 'webcam':
            self.fn = self.getCam
        else:
            raise ValueError("Invalid Type of source: %s"%(source))

    def get(self):
        #Common generator
        for img in self.fn():
            yield img

    def preprocess(self, image):
        image = resize_image(image)
        return image

    def getVideo(self):
        raise ValueError( "Not implemented!")

    def getImage(self):
        for idx in range(1,9):
            t0 = time.time()
            
            raw_image = Image.open('./samples/%d.jpg'%idx)
            t1 = time.time()
            print('Image Read time: ', t1- t0)

            #preprocess image
            raw_image = self.preprocess(raw_image)
            t2 = time.time()
            print('Image pre-process time: ', t2- t1)

            yield raw_image 
            
    def getClient(self):
        raise ValueError( "Not implemented!")

    def getCam(self):
        raise ValueError( "Not implemented!")


    
class DataOut:
    def __init__(self, dest):
        if dest == 'console':
            self.fn = self.writeConsole
        elif dest == 'client':
            self.fn = self.sendClient
        else:
            raise ValueError("Invalid Type of destination: %s"%(dest))

    def send(self, msg):
        self.fn(msg)

    def writeConsole(self, msg):        
        print("Caption: %s"%(msg))

    def sendClient(self):
        raise ValueError("Not implemented!")
