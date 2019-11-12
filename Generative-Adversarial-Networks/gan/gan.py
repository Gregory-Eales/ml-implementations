import torch

from discriminator import Discriminator
from generator import Generator



class GAN(object):

    def __init__(self):

        self.generator = Generator()
        self.discriminator = Discriminator()

    def train(self):
        pass
