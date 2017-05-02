import numpy as np
import operator as op
import time
class NaiveBayes:
    def __init__(self):
        self.stat=0
    def loadfile(self,filename):
         