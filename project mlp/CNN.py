#!/usr/bin/env python
#-*- coding: latin-1 -*-
import numpy as np
from scipy.signal import convolve2d
from outils import sousEchantillonner


#cette classe gere la couche convolution-sous echantillonnage 
class CNN:
    #filtresPoids : tenseur contenant les poids des neurones charger de la convolution(filtrage)
    #nbrFilres : le nombre de filtresPoids
    #filreBais : vecteur regroupant les bias de ses neurones de filtrage
    #dimSech : pas du sous echantillonnage
    #sechrsPoids : le poids des neurones de sous echantillonnage
    #sechrBias : le bias des neurones de sous echantillonnage
    def __init__(self,dimFiltres,dimSech=(2,2),couchePrec=None,coucheSucc=None):
        assert isinstance(dimFiltres,tuple) and isinstance(dimSech,tuple)
        assert len(dimSech) == 2 and len(dimFiltres) == 4
    #on initialise aléatoirement les poids des filtres    
        fan_in=np.prod(dimFiltres[1:])
        fan_out=dimFiltres[0]*np.prod(dimFiltres[2:])/np.prod(dimSech)
        w_bound=np.sqrt(6./(fan_in+fan_out))
        self.filtresPoids=2*w_bound*np.random.rand(*dimFiltres)-w_bound
        
        self.dimFiltres=dimFiltres
        self.deltaPoids=np.zeros(dimFiltres)
        self.filtresBias=np.zeros(dimFiltres[0])
        self.dimSech=dimSech
        self.couchePrec=couchePrec
        self.coucheSucc=coucheSucc
        self.dimInput   =  None
        self.dimConv    =  None
        self.dimOutput  =  None
        self.convOut    =  None
        self.output     =  None
 
#calcule des sorties
    def calculerOutput(self,images,nbrImages):
        imagesSech=np.zeros((nbrImages,)+self.dimOutput)
        imagesConv=np.zeros((nbrImages,)+self.dimConv)
        for j in range(self.dimFiltres[0]):
            for l in range(nbrImages):
                #operation de convolution 
                for i in range(self.dimInput[0]):
                    imagesConv[l,j]+=convolve2d(images[l,i],self.filtresPoids[j,i],'valid')
                #activation
                imagesConv[l,j]=self.activer(self.filtresBias[j]+imagesConv[l,j])
                #sous echantillonnage
                imagesSech[l,j]=sousEchantillonner(imagesConv[l,j],self.dimSech)
        return imagesConv, imagesSech 
    
    def activer(self,z):
        return 1/(1+np.exp(-z))
    
    def getCoutPoids(self):
        return (self.filtresPoids**2).sum()
    
    def alimenter(self,input):
        dimInput=np.shape(input)
        if self.dimInput is None:
            assert len(dimInput) >=3
            self.dimInput=np.shape(input)[1:]
            
            assert self.dimInput[-2] >=  self.dimFiltres[-2] and self.dimInput[-1] >= self.dimFiltres[-1]
            self.dimConv=(self.dimFiltres[0],)+(self.dimInput[-2]-self.dimFiltres[-2]+1,self.dimInput[-1]-self.dimFiltres[-1]+1)
                
            assert self.dimConv[-2]%self.dimSech[-2]==0 and self.dimConv[-1]%self.dimSech[-1]==0 
            self.dimOutput=(self.dimFiltres[0],)+(self.dimConv[-2]/self.dimSech[-2],self.dimConv[-1]/self.dimSech[-1])
          
        self.input=input.reshape((dimInput[0],)+self.dimInput)
        self.convOut,self.output=self.calculerOutput(self.input,np.shape(input)[0])
            
        if self.coucheSucc is not None:
            self.coucheSucc.alimenter(self.output)
    
    #propagation d'erreur 
    def propagerErreur(self,deltaOut,nu,moment,poidDecay,targets=None):
        nbrImages=np.shape(deltaOut)[0]
        deltaOut=deltaOut.reshape((nbrImages,)+self.dimOutput)
        deltaConv=np.zeros((nbrImages,)+self.dimConv)

        for l in range(nbrImages):
            for j in range(self.dimFiltres[0]):
                #calcul du vecteur d'erreur delta pour la couche de convolution
                deltaConv[l,j]=self.convOut[l,j]*(1.-self.convOut[l,j])*np.kron(deltaOut[l,j],(1./np.prod(self.dimSech))*np.ones(self.dimSech))
                #calcule du gradient
                for i in range(self.dimFiltres[1]):
                    if( self.couchePrec is not None ):
                        self.deltaPoids[j,i]=-nu*(convolve2d(self.input[l,i],deltaConv[l,j],'valid')
                    +(poidDecay/nbrImages)*self.filtresPoids[j,i])+moment*self.deltaPoids[j,i]
                    
        if (self.couchePrec is not None):
            deltaInput=np.zeros((nbrImages,)+self.dimInput)
            for l in range(nbrImages):
                for j in range(self.dimFiltres[0]):
                    for i in range(self.dimFiltres[1]):
                        deltaInput[l,i]+=convolve2d(deltaConv[l,j],self.filtresPoids[j,i],'full')
            self.couchePrec.propagerErreur(deltaInput,nu,moment,poidDecay)
        #actualisation des paramètres    
        self.filtresBias+=-nu*deltaConv.sum(axis=(0,2,3))
        self.filtresPoids+=self.deltaPoids


            

        

        
        