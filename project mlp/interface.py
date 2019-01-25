#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PySide import QtGui,QtCore
from scipy.signal import convolve2d
import numpy as np
from PIL import Image
from outils import *
import Image
from Reseau import Reseau
from MLP import MLP
from CNN import CNN

class ZoneDessin(QtGui.QWidget) :
	def __init__(self,parent=None) :
		QtGui.QWidget.__init__(self, parent)
		self.im=QtGui.QPixmap(28*5,28*5)
		self.im.fill(QtCore.Qt.white)
		self.x, self.y=0,0
		
	def mouseMoveEvent(self,e):
		path = QtGui.QPainterPath()
		path.moveTo(self.x, self.y)
		path.lineTo(e.x(), e.y())
		p=QtGui.QPainter(self.im)
		p.setPen(QtGui.QPen(QtCore.Qt.black, 10, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
		p.drawPath(path)
		self.repaint()
		self.x, self.y = e.x() , e.y()

	def mousePressEvent (self,e):
		self.x, self.y = e.x() , e.y()
	
	def paintEvent(self,e) :
		p=QtGui.QPainter(self)
		p.drawPixmap(0,0,self.im)  
		
	def toBinaryMatrix(self):
		myimage = self.im.toImage()    
		width = myimage.width()
		height = myimage.height()
		matrix = np.zeros((width,height)) 
		for i in range(height):
			for j in range(width):
				matrix[i,j] = 1.-QtGui.qGray(myimage.pixel(j,i))/255
		filtre=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
		matrix=convolve2d(matrix,filtre)
		return sousEchantillonner(matrix,(5,5),'mean')

class MaFenetre(QtGui.QMainWindow) :
	def __init__(self, parent=None) :
		QtGui.QMainWindow.__init__(self, parent)
		self.setWindowTitle("reseau neurones test")
		self.resize(420,420)
		self.dessin=ZoneDessin(self)
		self.dessin.setGeometry(50,50,28*5,28*5)
		self.btn_0 = QtGui.QPushButton("tester", self) 
		self.btn_0.clicked.connect(self.tester)
		self.btn_0.setGeometry(50, 280,150,50)
		self.btn_1 = QtGui.QPushButton("effacer", self) 
		self.btn_1.clicked.connect(self.effacer)
		self.btn_1.setGeometry(50, 340,150,50)
		self.btn_2 = QtGui.QPushButton("charger un reseau", self) 
		self.btn_2.clicked.connect(self.charger)
		self.btn_2.setGeometry(50, 220,150,50)
		self.btn_3 = QtGui.QPushButton("quitter", self) 
		self.btn_3.clicked.connect(QtCore.QCoreApplication.instance().quit)
		self.btn_3.setGeometry(300, 340,100,40)
		self.label = QtGui.QLabel(self)
		self.label.setGeometry(200,115,175,75)
		self.label.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
		self.label.setText("reponse du reseau : ?\n\n")
		self.label.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
		self.reseau=None
	
	def effacer(self):

		self.dessin.im.fill(QtCore.Qt.white)
		self.dessin.repaint()
		self.label.setText("reponse du reseau : ?\n\n")
		
	def tester(self):
		if (self.reseau==None):
			QtGui.QMessageBox.information(self, "Information", "Vouds devez d'abord charger un reseau")
		else :
			input  = self.dessin.toBinaryMatrix()
			output = self.reseau.getOutput(input.reshape((1,)+np.shape(input)))
			classe = np.argmax(output)
			self.label.setText("reponse du reseau : {}\n\n".format(classe))

		
	def charger(self):
		nomfichier ,_ = QtGui.QFileDialog.getOpenFileName(self,
		str("charger un reseau"), "/home/jana", str("reseau sauvegarde (*.txt *.csv)"))
		if nomfichier != '':			
			self.reseau=Reseau()
			self.reseau.charger(nomfichier)
			if isinstance(self.reseau.couches[0],CNN):
				self.dessin.setGeometry(50,50,self.reseau.couches[0].dimInput[1]*5+2,self.reseau.couches[0].dimInput[2]*5+2)
			if isinstance(self.reseau.couches[0],MLP):
				self.dessin.setGeometry(*(50,50)+(int(np.sqrt(self.reseau.couches[0].dimInput))*5+2,)*2)
				



app = QtGui.QApplication(sys.argv)
f = MaFenetre()
f.show()
sys.exit(app.exec_())

