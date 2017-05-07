from __future__ import print_function
import numpy as np
import time
from PIL import Image
from resize import resize_image

import io
import socket
import struct

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
        for idx in range(1,10):
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

        # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means # all interfaces)
        server_socket = socket.socket()
        server_socket.bind(('192.168.43.61', 8000))
        server_socket.listen(0)

        print ("Server Ready. Waiting for incoming data...")
        # Accept a single connection and make a file-like object out of it
        connection = server_socket.accept()[0].makefile('rb')
        ctr = 0
        try:
            while True:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    raise ValueError("Connection Problem: Cannot receive correct length of Data")
                    break
                
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(connection.read(image_len))
                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)
                image = Image.open(image_stream)
                
                #saveStr = "a" + str(ctr) + ".jpg"
                #image.save(saveStr)
                #ctr += 1

                #server_socket.send(saveStr.encode('ascii'))
               
                #preprocess image
                t2 = time.time()
                image = self.preprocess(image)
                print('Image pre-process time: ', time.time()-t2)

                print('Image is %dx%d' % image.size)
                #image.verify()
                #print('Image is verified')

                yield image

        finally:
            connection.close()
            server_socket.close()

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
