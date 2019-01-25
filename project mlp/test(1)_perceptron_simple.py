#!/usr/bin/env python
#-*- coding: latin-1 -*-

from outils import *
from mnistHandwriting import *
from Reseau import Reseau
from CNN import CNN
from MLP import MLP
from matplotlib import pyplot as plt

##test pour un perceptron sur deux nuages gaussiens separees
#la base d'aprentissage serait des sous echantillons de deux nuages gaussiens le premier centre sur (2,2)
#et le deuxieme sur (2,6)
nbrExpl = 500
x,y=generNuageGaussien(((2,2),(2,6)),nbrExpl)

#on affiche la base d'aprentissage (la une couleur designe la classe)
couleur={0:'r',1:'b'}
for k in range(nbrExpl):
    plt.plot(x[k,0],x[k,1],'ro',color=couleur[y[k]])
plt.show()


#on initialise les parametres du perceptron
nu=0.1
Nin = 2


#algotithme du gradient descendant stochastique
Niter=100
couts=np.zeros(Niter)
w=np.random.rand(Nin)-.5
b=0
sorties=np.zeros(nbrExpl)
for numIter in range(Niter):
    cout=0
    for numExpl in range(nbrExpl):
        y_chapeau = sigmoid(np.dot(w,x[numExpl])+b)
        sorties[numExpl]=y_chapeau
        cout+=.5*(y_chapeau-y[numExpl])**2
        w=w-nu*(y_chapeau-y[numExpl])*y_chapeau*(1-y_chapeau)*x[numExpl]
        b=b-nu*(y_chapeau-y[numExpl])*y_chapeau*(1-y_chapeau)
    couts[numIter]=(cout/nbrExpl)

#on afficher la figure de variation du cout par iteration
fig ,ax  = plt.subplots()
plt.plot(couts)
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout ")
plt.show()

#on teste les performances du perceptron sur une base de test
nbrExplTest=50
tauxErreur=0
x,y=generNuageGaussien(((2,2),(2,6)),nbrExplTest)
reponses=np.zeros(nbrExplTest)
for numExpl in range(nbrExplTest):
    y_chapeau = sigmoid(np.dot(w,x[numExpl])+b)
    y_chapeau = np.argmin((np.array([0,1])-y_chapeau)**2)
    reponses[numExpl]=y_chapeau
    tauxErreur+=(y_chapeau!=y[numExpl])
#on affiche la representation spatiale des elements de la base de test
for k in range(nbrExplTest):
    plt.plot(x[k,0],x[k,1],'ro',color=couleur[y[k]])
plt.show()
#on reaffiche la representation des éléments de la base de test mais cette fois etiqueté par les
#predictions du réseau 
for k in range(nbrExplTest):
    plt.plot(x[k,0],x[k,1],'ro',color=couleur[reponses[k]])
plt.show()
#on normalise le taux et on l'affiche
tauxErreur=100*tauxErreur/nbrExplTest
print "le taux d'erreur sur la base de test est {}\n".format(tauxErreur)
    

    



