import numpy as np
from sklearn.neural_network import MLPRegressor
import os
import pandas as pd
from cv2 import * 


x_train=[]
y_train=[]
train_file='train.csv'
root='/home/shashanks/git_code/mapmy_india/Train/'
root_test='/home/shashanks/git_code/mapmy_india/Test/'

def prepare_training(train_file,root):
	'''
	Preparing Training Data from images
	'''

	global x_train,y_train
	df=pd.read_csv(train_file)

	for index,val in df.iterrows():

		image=df.loc[index]['Img_Name']
		label=df.loc[index]['Label']
		y=df.loc[index]['Top']
		x=df.loc[index]['Left']
		w=df.loc[index]['Width']
		h=df.loc[index]['Height']
		img=cv2.imread(root+image)
		img.resize(32,16)
		img=img.flatten()
		x_train.append(img)
		y=[y,x,w,h]
		y_train.append(y)
	
	return np.array(x_train),np.array(y_train)

def make_model(train_file,root):
	'''
	Building Model
	'''

	x_train,y_train=prepare_training(train_file,root)
	print(x_train.shape)
	print(y_train.shape)
	model=MLPRegressor(hidden_layer_sizes=(8,),activation='relu',solver='lbfgs',learning_rate='adaptive',max_iter=100000, learning_rate_init=0.01,alpha=0.01)
	model.fit(x_train,y_train)
	return model


def prepare_test(root_test):
	'''
	Prepare Test Data from images
	'''
	x_test=[]

	for image in os.listdir(root_test):
		
		img=cv2.imread(root_test+image)
		img.resize(32,16)
		img=img.flatten()
		x_test.append(img)

	return np.array(x_test)


#Build Model and predict for test images

model=make_model(train_file,root)	
x_test=prepare_test(root_test)
print('x_test shape',x_test.shape)


result=model.predict(x_test)


plist=os.listdir(root_test)
result=result.tolist()
for row in result:
	index=result.index(row)
	img=cv2.imread(root_test+plist[index])
	yt=int(round(row[0]))
	xt=int(round(row[1]))
	wt=int(round(row[2]))
	ht=int(round(row[3]))
	img=cv2.rectangle(img, (xt, yt), (xt+wt, yt+ht), (0, 255, 0), 2)
	cv2.imshow('result',img)
	cv2.waitKey(1000)
	
print("training score",model.score(x_train,y_train))
