#!/usr/bin/env python
#-*- coding: latin-1 -*-

import numpy as np

from outils import *
from mnistHandwriting import *
from Reseau import Reseau
from CNN import CNN
from MLP import MLP

#parametres du reseau

nu                  = 1e-1
nbrIter             = 400
nbrExplApr          = 2560
nbrExpltest         = 2560
paquetSize          = 32
moment              = 0.0
poidDecay           = 0.0008
nbrPaquets=nbrExplApr/paquetSize


baseApr=chargerBase('data/digit10_28x28_learn.txt')
images , targets = adaptData(baseApr,nbrExplApr,(28,28))
reseau=Reseau(nu,moment,poidDecay)
reseau.addCouche(MLP((28*28,120,10)))
reseau.entrainer(images,targets,nbrIter,paquetSize,nbrPaquets)

basetest=MNISTexample(0,nbrExpltest,only01=False,bTrain=False)
#basetest=chargerBase('data/digit10_28x28_test.txt')
images,targets=adaptData(basetest,nbrExpltest,(28,28))
reseau.tester(images,targets,nbrExpltest)        
