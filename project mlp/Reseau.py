#!/usr/bin/env python
#-*- coding: latin-1 -*-
from MLP import MLP
from CNN import CNN
import numpy as np

class Reseau:
    def __init__(self ,learning_rate=1e-1,momentum=0,weights_decay=0):
        self.nu=learning_rate
        self.moment=momentum
        self.poidDecay=weights_decay
        self.couches=None
        
    def addCouche(self,couche):
        assert isinstance(couche,MLP) or isinstance(couche,CNN)
        if self.couches is None :
            self.couches=[couche]
        else :
            self.couches.append(couche)
            self.couches[-1].couchePrec=self.couches[-2]
            self.couches[-2].coucheSucc=self.couches[-1]

            
    def entrainer(self,input,targets,nbrIter,nbrPaquets,paquetSize=1,cout_type='quadratique',getCouts=False,autosave=None):
        print 'entrainement du reseau en cours...'
        if getCouts:
            couts=np.zeros(nbrIter)
        for k in range(nbrIter):
            if getCouts:
                cout=0
            if autosave is not None:
                if (k+1)%autosave==0 :
                    self.sauvegarder('sauvegarde/sauvegarde{}.csv'.format(k))
                    
            for l in range(nbrPaquets):
                    targetsPaquet=targets[l*paquetSize:(l+1)*paquetSize]
                    inputPaquet=input[l*paquetSize:(l+1)*paquetSize]
                    #alimentation de la premiere couche qui de sa part va propager sa sortie vers la couche suivante
                    #ainsi de suite
                    self.couches[0].alimenter(inputPaquet)
                    if getCouts:
                        cout+=self.calculerCout(self.couches[-1].output[-1],targetsPaquet,paquetSize*nbrPaquets,cout_type)
                    self.couches[-1].propagerErreur(None,self.nu,self.moment,self.poidDecay,targetsPaquet,cout_type)
            if getCouts:
                couts[k]=cout
            print couts[k]
        print 'entrainement terminer avec succes'
        if getCouts:
            return couts
        
    def calculerCout(self,outputs,targets,N,cout_type='quadratique'):
        cout=0
        assert cout_type=='quadratique' or cout_type=='crossEntropy' 
        for couche in self.couches :
            cout+=(.5/N)*self.poidDecay*couche.getCoutPoids()
        if cout_type=='quadratique':
            cout += (.5/N)*np.linalg.norm(outputs-targets)**2
        if cout_type=='crossEntropy':
            cout+=-(1./N)*np.nan_to_num(((targets*np.log(outputs)+(1.-targets)*np.log(1.-outputs)).sum()))
 
        return cout
    
    def tester(self,input,targets,nbrExplTest,getmatrice=False,getoutput=False):
        assert len(np.shape(targets))==2 and np.shape(targets)[1] == self.couches[-1].nbrClasses
        taux=0
        
        self.couches[0].alimenter(input)
        outputs=self.couches[-1].output[-1]
        
        xout=np.zeros((nbrExplTest,self.couches[-1].nbrClasses))
            
        for k in range(nbrExplTest):
                xout[k,np.argmax(outputs[k])]=1
            
        taux=100*(1-(xout*targets).sum()/nbrExplTest)
        
        if getmatrice:
            matrice=np.zeros((self.couches[-1].nbrClasses,self.couches[-1].nbrClasses))
            matrice=np.dot(np.rot90(np.fliplr(targets)),xout)
            return taux , matrice ,xout
        
        return taux , xout
    
    def sauvegarder(self,nomfichier):
        fichier=open(nomfichier,'w')
        for couche in self.couches:
            if isinstance(couche,MLP):
                fichier.write('MLP\n')
                fichier.write(repr(couche.nbrElmCouche))
                fichier.write('\n')
                for k in range(couche.nbrCouches):
                    fichier.write(' '.join(repr(x) for x in couche.poids[k].flatten()))
                    fichier.write('\n')
                    fichier.write(' '.join(repr(x) for x in couche.bias[k]))
                    fichier.write('\n')
            elif isinstance(couche,CNN):
                fichier.write('CNN\n')
                fichier.write(repr(couche.dimFiltres))
                fichier.write('\n')
                fichier.write(repr(couche.dimSech))
                fichier.write('\n')
                fichier.write(repr(couche.dimInput))
                fichier.write('\n')
                fichier.write(' '.join(repr(x) for x in couche.filtresPoids.flatten()))
                fichier.write('\n')
                fichier.write(' '.join(repr(x) for x in couche.filtresBias.flatten()))
                fichier.write('\n')
        fichier.write('end')
        fichier.close()
 
    def charger(self,nomfichier):
        from ast import literal_eval
        self.couches=None
        fichier=open(nomfichier,'r')
        couche_type = fichier.readline().rstrip('\n\r')
        while couche_type != 'end':
            if couche_type == 'MLP':
                nbrElmCouche=literal_eval(fichier.readline().rstrip('\n\r'))
                couche=MLP(nbrElmCouche)
                for k in range(couche.nbrCouches):
                    poids=fichier.readline().rstrip('\n\r').split(" ")
                    couche.poids[k]=np.array(map(float,poids)).reshape(nbrElmCouche[k],nbrElmCouche[k+1])
                    bias=fichier.readline().rstrip('\n\r').split(" ")
                    couche.bias[k]=np.array(map(float,bias))
            elif couche_type == 'CNN':
                dimFiltres=literal_eval(fichier.readline().rstrip('\n\r'))
                dimSech=literal_eval(fichier.readline().rstrip('\n\r'))
                couche=CNN(dimFiltres,dimSech)
                couche.dimInput=literal_eval(fichier.readline().rstrip('\n\r'))
                couche.filtresPoids=fichier.readline().rstrip('\n\r').split(" ")
                couche.filtresPoids=np.array(map(float,couche.filtresPoids)).reshape(dimFiltres)
                couche.filtresBias=fichier.readline().rstrip('\n\r').split(" ")
                couche.filtresBias=np.array(map(float,couche.filtresBias)).reshape(dimFiltres[0])
                couche.dimConv=(couche.dimFiltres[0],)+(couche.dimInput[-2]-couche.dimFiltres[-2]+1,couche.dimInput[-1]-couche.dimFiltres[-1]+1)
                couche.dimOutput=(couche.dimFiltres[0],)+(couche.dimConv[-2]/couche.dimSech[-2],couche.dimConv[-1]/couche.dimSech[-1])
            self.addCouche(couche)
            couche_type= fichier.readline().rstrip('\n\r')
        
    def getOutput(self,Xin):
        self.couches[0].alimenter(Xin)
        return self.couches[-1].output[-1]
        

    
        

        