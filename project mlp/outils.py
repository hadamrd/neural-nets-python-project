#!/usr/bin/env python
#-*- coding: latin-1 -*-
 
import numpy as np


#une autre fonction qui permet de réduire la dimension des bases
#cette fonction en blocs le vecteur Xin puis le transforme en matrice y
#après pour chaque bloc on calcule deux valeurs x1 et x2 qu'on range dans le vecteur Xinnew
#x1 est le nombre de diagonale du block qui contiennent 1
#x2 est le nombre de antidiagonales du bloc qui contiennent 1
def acrossCornerExtract(Xin):
    Xinnew=[]
    dim=int(np.sqrt(len(Xin)))
    y=np.array(Xin).reshape(dim,dim)
    y=isolate(y)
    size=len(y)
    blocSize=size/4
    for i in range(4):
        for j in range(4):
            x1=0
            x2=0
            borderi=(i==6 and size) or (i+1)*blocSize
            borderj=(j==6 and size) or (j+1)*blocSize
            bloc=y[blocSize*i:borderi,blocSize*j:borderj]
            for k in range(-len(bloc)+1,len(bloc)):
                x1+=np.diag(bloc,k).any()
                x2+=np.diag(np.rot90(np.rot90(bloc)),k).any()
            Xinnew+=[x1,x2]
    return Xinnew

def sonde(Xin):
    bl=4
    Xinnew=[]
    y=np.array(Xin).reshape(28,28)
    for i in range:
        code
    
    

#cette fonction permet de supprimer les bords d'une image binaire (0 , 1)
def isolate(matrice):
    for k in range(len(matrice)):
        if(matrice[k].any()):
            matrice=matrice[k:]
            break
    for k in range(len(matrice[0])):
        if(matrice[:,k].any()):
            matrice=matrice[:,k:]
            break
    for k in range(len(matrice)):
        if(matrice[len(matrice)-1-k].any()):
            matrice=matrice[:len(matrice)-k]
            break
    for k in range(len(matrice[0])):
        if(matrice[:,len(matrice[0])-1-k].any()):
            matrice=matrice[:,:len(matrice[0])-k]
            break
    return matrice

def eqmtest(reseauN,base):
    EQM=0
    for d in base.Test:
        Xout=np.argmax(reseauN.output(d[:-1]))
        Target=self.classeVect(d[-1])
        EQM+=0.5*np.inner(Target-Xout,Target-Xout)
    return EQM
    



def sousEchantillonner(image,dimSech,type='mean'):
    dimImage=np.shape(image)
    dimImageSech=(dimImage[0]/dimSech[0],dimImage[1]/dimSech[1])
    imgSech=np.zeros(dimImageSech)
    for k in range(dimImageSech[0]):
        for l in range(dimImageSech[1]):
            if type == 'mean':
                imgSech[k,l]=(1./np.prod(dimSech))*image[k*dimSech[0]:(k+1)*dimSech[0],l*dimSech[1]:(l+1)*dimSech[1]].sum()
            elif type == 'max':
                imgSech[k,l]=image[k*dimSech[0]:(k+1)*dimSech[0],l*dimSech[1]:(l+1)*dimSech[1]].max()
    return imgSech

def generNuageGaussien(T,n,pour='perceptron',nomFichier=None):
    numClasse=0
    nbrClasse=len(T)
    dim=(n,len(T[0]))
    if nomFichier is not None:
        f=open(nomFichier,'w')
        f.write(str(nbrClasse)+"_nuages gaussiens\n")
        f.write(repr(dim))
        f.write(nbrClasse)  
    nuage=np.zeros((n,2))
    if pour == 'perceptron':
        etiquete=np.zeros(n)
    elif pour == 'mlp':
        etiquete =np.zeros((n,nbrClasse))
    for x, y in T:
        nuage[numClasse*n/nbrClasse:(numClasse+1)*n/nbrClasse]=np.append(np.random.randn(n/nbrClasse,1)+
                                                                         x,np.random.randn(n/nbrClasse,1)+y,1)
        if pour == 'perceptron':
                etiquete[numClasse*n/nbrClasse:(numClasse+1)*n/nbrClasse]=numClasse*np.ones(n/nbrClasse)
        elif pour == 'mlp':
            etiquete[numClasse*n/nbrClasse:(numClasse+1)*n/nbrClasse,numClasse]=np.ones(n/nbrClasse)
        if nomFichier is not None :
            for k in range(n) :
                f.write(' '.join(repr(x) for x in nuage[k]))
                f.write(' {}\n'.format(etiquete[k]))
        numClasse+=1
    if nomFichier is not None:
        f.close()
    else :
        return nuage,etiquete
        
def sigmoid(z):
    return 1./(1.+np.exp(-z))

    
def chargerBase(nomfichier):
    from ast import literal_eval
    fichier=open(nomfichier,'r')
    entete=fichier.readline().rstrip('\r\n')
    print (entete)
    dim=literal_eval(fichier.readline().rstrip('\r\n'))
    print 'dimension de la base : '+repr(dim)
    nbrClass=int(fichier.readline().rstrip('\r\n'))
    base=np.zeros(dim)
    targets=np.zeros((dim[0],nbrClass))
    print 'chargement de la base en cours'
    for k in range(dim[0]):    
        l=fichier.readline().rstrip('\r\n').split(" ")
        if  ( l == ['']  ):
            l=fichier.readline().rstrip('\r\n').split(" ")

        if ( l[-1] == '\r\n' or l[-1] == '') :
            l=l[:-1]
        
        base[k,:]=np.array(map(float,l[:-1]))
        targets[k,:]=classetoVect(float(l[-1]),nbrClass)

    print 'chargement termine.'
    return base , targets
    
def classetoVect(indice,longeur):
    Target=np.zeros(longeur)
    Target[int(indice)]=1
    return Target


