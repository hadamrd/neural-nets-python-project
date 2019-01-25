#!/usr/bin/env python
#-*- coding: latin-1 -*-

from Reseau import Reseau
from MLP import MLP
from CNN import CNN
from outils import *
from matplotlib import pyplot as plt

nbrExplApr = 783
Niter = 100
paquetSize =19
nbrPaquet = nbrExplApr/paquetSize

#creation du modele
reseau=Reseau(learning_rate=0.1,momentum=0.8)
reseau.addCouche(CNN((4,1,5,5)))
reseau.addCouche(MLP((4*12*12,120,10)))

#chargement de la base d'apprentissage
images , etiquetes = chargerBase('data/digit10_28x28_learn.txt')
images=images.reshape(nbrExplApr,1,28,28)

#entrainement du modele

couts=reseau.entrainer(images,etiquetes,Niter,nbrPaquet,paquetSize,cout_type='crossEntropy',getCouts=True,autosave=20)

#affichage du cout par iteration

plt.plot(couts,label="aprentissage")
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout ")
plt.show()
#teste des performances

#chargement de la base de test
images , etiquetes  = chargerBase('data/digit10_28x28_test.txt')
nbrExplTest = len(images)
images=images.reshape(nbrExplTest,1,28,28)
taux , matrice , xout= reseau.tester(images,etiquetes,nbrExplTest,getmatrice=True)

print "le taux d'erreur sur la base d'aprentissage est : " + str(taux)
print matrice 

