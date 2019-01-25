
from Reseau import Reseau
from MLP import MLP
from outils import *
from matplotlib import pyplot as plt
nbrEntrees = 28*28
nbrClasses = 10
tailleCoucheC = 120

#creation du modele
reseau=Reseau(learning_rate=0.1)
reseau.addCouche(MLP((nbrEntrees,tailleCoucheC,nbrClasses)))

#chargement de la base d'apprentissage
images , etiquetes = chargerBase('data/digit10_28x28_learn.txt')
nbrExpl = 783
#entrainement du modele
Niter = 200
paquetSize =19
nbrPaquet = 783/paquetSize
couts=reseau.entrainer(images,etiquetes,Niter,nbrPaquet,paquetSize,cout_type='quadratique',getCouts=True,autosave=1)

#affichage du cout par iteration

plt.plot(couts,label="aprentissage")
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout ")

#teste des performances

#chargement de la base de test
images , etiquetes  = chargerBase('data/digit10_28x28_test.txt')
nbrExpl = len(images)
coutsTest=np.zeros(Niter)

for k in range(Niter):
    reseau.charger('sauvegarde/sauvegarde{}.csv'.format(k))    
    reseau.couches[0].alimenter(images)
    outputs=reseau.couches[-1].output[-1]
    coutsTest[k]=reseau.calculerCout(outputs,etiquetes,nbrExpl)

plt.plot(coutsTest,label="test")
plt.legend(loc='upper right')    

plt.show()
taux , matrice , xout= reseau.tester(images,etiquetes,nbrExpl,getmatrice=True)

print taux
print matrice 
