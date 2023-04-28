from pynq import DefaultHierarchy
from pynq import allocate
import numpy as np
import time

class MNISTDriver(DefaultHierarchy):
    def __init__(self, description):
        super().__init__(description=description)
        CONTROL_REGISTER = 0
        print("Initial control register: ", self.mnist_dnn.read(CONTROL_REGISTER))
        self.mnist_dnn.write(CONTROL_REGISTER, self.mnist_dnn.read(CONTROL_REGISTER) | 0x81)
        print("Updated control register: ", self.mnist_dnn.read(CONTROL_REGISTER))
        self.dma = self.axi_dma_0
        self.dma_send = self.dma.sendchannel
        self.dma_recv = self.dma.recvchannel

    def inference(self, img, result):
        
        row_img = len(img)
        col_img = len(img[0])
        
        # Ensure A and B are the right shape
        if ((row_img != 28) or (col_img != 28)):
            raise Exception("Input matrix dimensions do not match the spec!")
            
        # Flatten the matrix
        # This operation probably takes a lot of time, so try not to use THIS function
        combined = allocate(shape=((row_img * col_img,)), dtype=np.uint8)
        index = 0
        for row in img:
            for entry in row:
                combined[index] = entry
                index = index + 1
        
        # Initialize the DMA transfer to the IP
        self.dma_send.transfer(combined)
        
        # Once data was sent, wait for data to come in
        self.dma_recv.transfer(result)
        self.dma_recv.wait()
               
        # Free the allocated memory                    
        del combined
        
        return
    
    # This function requires the memory spaces to be pre-allocated, but it should be much faster.
    def batch_inference(self, flattened_images, results):
    
        transfer_length_send = flattened_images[0].shape[0] * flattened_images[0].itemsize
        transfer_length_recv = results[0].shape[0] * results[0].itemsize

        for itr, image in enumerate(flattened_images):
            self.dma_send.transfer(image, nbytes=transfer_length_send)
            self.dma_send.wait() 
            self.dma_recv.transfer(results[itr], nbytes=transfer_length_recv)
            self.dma_recv.wait() 
        return
    
    @staticmethod
    def checkhierarchy(description):
        if ('axi_dma_0' in description['ip'] and 'mnist_dnn' in description['ip']):
            return True
        else:
            return False
        return True
