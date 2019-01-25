#!/usr/bin/env python
#-*- coding: latin-1 -*-

import numpy as np
#MLP Multi Layer perceptron
class MLP:
    #nbrElmCouche : liste contenant le nombre de neurones de chaque couche
    #nbrElmChouche[0] est le nombre d'entrées
    #nbrCouches : le nombre de couches
    #nbrElmCouche[-1] est le nombre de calsses à classifier 
    #poids : une liste de matrices contenant les poids des neurones  de chaque couche 
    #activer : la fonction de seuil qu'appliquera chaque neurone a sa sortie ,expl sigmoid ,tangh...
    def __init__(self,nbrElmCouche,couchePrec=None,coucheSucc=None):
        
        assert isinstance(nbrElmCouche, tuple ) or isinstance(nbrElmCouche,list) and len(nbrElmCouche)>1
        
        self.nbrElmCouche=nbrElmCouche
        self.nbrCouches=len(nbrElmCouche)-1
        self.poids=[]
        self.deltaPoids=[]
        #on initialise les poids de chaque couche selon une distribution uniforme sur l'interval [-r,r]  
        for k in range(self.nbrCouches):
            r=np.sqrt(6)/np.sqrt(nbrElmCouche[k]+nbrElmCouche[k+1])
            self.poids.append(2*r*np.random.rand(nbrElmCouche[k],nbrElmCouche[k+1])-r)
            self.deltaPoids.append(np.zeros((nbrElmCouche[k],nbrElmCouche[k+1])))
            
        self.coucheSucc=coucheSucc
        self.couchePrec=couchePrec
        #on initialise les seuils par zero
        self.bias =[np.zeros(k) for k in nbrElmCouche[1:]]
        self.input=None
        self.dimInput=nbrElmCouche[0]
        self.nbrClasses=nbrElmCouche[-1]
        self.output=None


    #la fonction d'activation dans notre cas sigmoide
    #on peut envisager d'ajouter d'autre foctions comme tanh
    def activer(self,x):
        return 1/(1+np.exp(-x))
    
    #cette méthode prend en entrée une base dedonnées et renvoie dans une liste
    #la sortie de chaque couche ,outputs[-1] est une matrice de taille numElmBase*nbrClasses
    #qui contient les sorties du réseau
    def calculerOutput(self,X):
        dimInput=np.shape(X)
        assert dimInput[1] == self.dimInput
        outputs=[X]
        for k in range(self.nbrCouches):
            y=np.dot(outputs[-1],self.poids[k])
            outputs.append(self.activer(self.bias[k]+y))
        return outputs
    
    #cette méthode permet de mettre le reseau en marche
    #elle actualise ses attributs et fait le calcule des sorties (output)
    def alimenter(self,input):
        dimInput=np.shape(input)
        assert len(dimInput) >1 or np.prod(dimInput[1:])==self.dimInput
        self.input=input.reshape(dimInput[0],self.dimInput)
        self.output=self.calculerOutput(self.input)
        if self.coucheSucc is not None :
            return self.coucheSucc.alimenter(self.output[-1])

    
    #cette méthode permet de propager l'erreur dans le reseau selon l'algorithme du gradient descendant stochastique
    #nu : est le taux d'aprentissage
    #poidsDecay et moment sont des paramètres d'aprentissages optionnel

    def propagerErreur(self,deltaOut,nu,moment=0,poidsDecay=0,targets=None,cout_type='quadratique'):

    #calcule de la fonction de cout et de l'erreur à la sortie du reseau

        if self.coucheSucc is None:
            
            assert np.shape(targets)[1] == self.nbrClasses
            N=len(targets)
            assert len(np.shape(targets)) == 2 and isinstance(cout_type,str)
            if cout_type == 'quadratique':
                delta=(self.output[-1]-targets)*self.output[-1]*(1.-self.output[-1])/N
            elif cout_type == 'crossEntropy':
                delta=(self.output[-1]-targets)/N
        else :
            N=len(deltaOut)
            delta=self.output[-1]*(1-self.output[-1])*deltaOut
        #calcul du gradient des poids et des seuils pour la dernière couche     
        self.bias[-1]+=-nu*(delta.sum(axis=0))
        self.deltaPoids[-1]=-nu*np.dot(np.rot90(np.fliplr(self.output[-2])),delta)-(poidsDecay/N)*self.poids[-1]+moment*self.deltaPoids[-1]
        #calcul du gradient des poids et des seuils pour les autres couches
        for k in range(self.nbrCouches-1):
            delta                 =  self.output[-k-2]*(1.-self.output[-k-2])*np.dot(delta,np.rot90(np.fliplr(self.poids[-k-1])))
            self.deltaPoids[-k-2] = -nu*np.dot(np.rot90(np.fliplr(self.output[-k-3])),delta)-(poidsDecay/N)*self.poids[-k-2]+moment*self.deltaPoids[-k-2]
            self.bias[-k-2]      += -nu*(delta.sum(axis=0))
        #si il y a une couche avant dans le reseau on propage l'erreur 
        if(self.couchePrec is not None):
            deltaInput=np.dot(delta,np.rot90(np.fliplr(self.poids[0])))
            self.couchePrec.propagerErreur(deltaInput,nu,moment,poidsDecay)
        #on actualise alors les poids    
        self.poids=[w+dw for w,dw in zip(self.poids,self.deltaPoids)]
        

#le coût des poids et utilisable pour la variante weight decay de l'algorithme 
    def getCoutPoids(self):
        return sum((w**2).sum() for w in self.poids)
    

        

        


    
            
            
