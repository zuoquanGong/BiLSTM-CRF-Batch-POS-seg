# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:46:28 2018

@author: zuoquan gong
"""
from optparse import OptionParser 
from config import Params
from Pythagoras_master import Pythagoras

if __name__=="__main__":
    
    parser = OptionParser()
        
    parser.add_option("--train", dest="trainFile",
                  default="",help="train dataset")

    parser.add_option("--dev", dest="devFile",
                  default="",help="dev dataset")

    parser.add_option("--test", dest="testFile",
                  default="",help="test dataset")


    (options, args) = parser.parse_args()
    
    
    params=Params()
    if options.trainFile!="":
        params.process_params.trainFile=options.trainFile
    if options.devFile!="":
        params.process_params.devFile=options.devFile
    if options.testFile!="":
        params.process_params.testFile=options.testFile
    master = Pythagoras(params)
    master.train()