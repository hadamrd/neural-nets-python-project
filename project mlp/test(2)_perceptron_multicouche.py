#!/usr/bin/env python
#-*- coding: latin-1 -*-
from outils import *
from mnistHandwriting import *
from Reseau import Reseau
from CNN import CNN
from MLP import MLP
from matplotlib import pyplot as plt

nbrExpl = 800
centre1=(-2,-2)
centre2=(-2, 2)
centre3=( 2,-2)
centre4=( 2, 2)
x,y=generNuageGaussien((centre1,centre2,centre3,centre4),nbrExpl,pour='mlp')

#on affiche la base d'aprentissage (la une couleur designe la classe)
couleur={0:'r',1:'b',2:'g',3:'k'}
for k in range(nbrExpl):
    plt.plot(x[k,0],x[k,1],'ro',color=couleur[np.argmax(y[k])])
plt.show()

#creation d'un perceptron multicouches a l'aide de la classe Reseau et la classe MLP
reseau=Reseau(learning_rate=0.1)
reseau.addCouche(MLP((2,4)))

#aprentissage du modele
Niter = 100
couts = reseau.entrainer(x,y,Niter,nbrExpl,getCouts=True,autosave=1)

#affichage du cout par iteration



plt.plot(couts,label="aprentissage")
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout ")

#teste des performances
nbrExpl = 800

x,y = generNuageGaussien((centre1,centre2,centre3,centre4),nbrExpl,pour='mlp')
coutsTest=np.zeros(Niter)

for k in range(Niter):
    reseau.charger('sauvegarde/sauvegarde{}.csv'.format(k))    
    reseau.couches[0].alimenter(x)
    y_chapeau=reseau.couches[-1].output[-1]
    coutsTest[k]=reseau.calculerCout(y_chapeau,y,nbrExpl)

plt.plot(coutsTest,label="test")
plt.legend(loc='upper right')    
plt.show()

taux , reponses =reseau.tester(x,y,nbrExpl,getoutput=True)
print "le taux d'erreur sur la base de test est : "+str(taux)
#on affiche la representation spatiale des elements de la base de test
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title("base de test")
ax2.set_title("classification du reseau")

for k in range(nbrExpl):
    ax1.plot(x[k,0],x[k,1],'ro',color=couleur[np.argmax(y[k])])

#on reaffiche la representation des ?l?ments de la base de test mais cette fois etiquet? par les
#predictions du r?seau 
for k in range(nbrExpl):
    ax2.plot(x[k,0],x[k,1],'ro',color=couleur[np.argmax(reponses[k])])
plt.show()
