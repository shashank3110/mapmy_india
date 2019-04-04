import numpy as np
from sklearn.neural_network import MLPRegressor
import os
import pandas as pd
from cv2 import * 
import tensorflow as tf
from tensorflow.keras import Sequential,models,Input,utils,layers,Model,applications,optimizers
#Img_Name,Top,Left,Width,Height,Label
#02-25 13.26.44.jpg,131,1687,163,163,Speed Limit 80
#01-05 10.15.27_2.jpg,248,422,17,24,Speed Limit 60
#speed_map={'Speed Limit 20':[1,0,0,0,0,0],'Speed Limit 30':[0,1,0,0,0,0],'Speed Limit 40':[0,0,1,0,0,0],'Speed Limit 50':[0,0,0,1,0,0],'Speed Limit 60':[0,0,0,0,1,0],
#'Speed Limit 80':[0,0,0,0,0,1]}
x_train=[]
y_train=[]
train_file='train.csv'
root='/home/shashanks/git_code/mapmy_india/Train/'
root_test='/home/shashanks/git_code/mapmy_india/Test/'
# y=248
# x=422
# w=17
# h=24
def prepare_training(train_file,root):
	global x_train,y_train
	df=pd.read_csv(train_file)
	#print(df['Label'])
	#images=df['Img_Name']
	for index,val in df.iterrows():
		image=df.loc[index]['Img_Name']
		label=df.loc[index]['Label']
		y=df.loc[index]['Top']
		x=df.loc[index]['Left']
		w=df.loc[index]['Width']
		h=df.loc[index]['Height']
		img=cv2.imread(root+image)
		#print(type(img))
		#img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		img.resize(32,16)
		img=img.flatten()
		x_train.append(img)
		#y_train.append(speed_map[label])
		y=[y,x,w,h]#,y,w,h]#[[x],[y],[w],[h]]#tf.reshape([x,y,w,h],[4,1])
		#y=y.flatten()
		y_train.append(y)
	#y_train=np.array(y_train).flatten()
	#print(x_train)
	#print(y_train)
	return np.array(x_train),np.array(y_train)
#x_train,y_train=prepare_training(train_file,root)
#print(x_train.shape)
def make_model(train_file,root):

	x_train,y_train=prepare_training(train_file,root)
	print(x_train.shape)
	print("#########################################################")
	print(y_train.shape)
	model=MLPRegressor(hidden_layer_sizes=(8,),activation='relu',solver='lbfgs',learning_rate='adaptive',max_iter=100000, learning_rate_init=0.01,alpha=0.01)
	

	
	model.fit(x_train,y_train)
	return model

model=make_model(train_file,root)
def prepare_test(root_test):
	x_test=[]
	#df=pd.read_csv(train_file)
	#print(df['Label'])
	#images=df['Img_Name']
	for image in os.listdir(root_test):
		
		img=cv2.imread(root_test+image)
		#print(type(img))
		#img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		img.resize(32,16)
		img=img.flatten()
		x_test.append(img)
		#y_train.append(speed_map[label])
	#print(x_train)
	#print(x_test)
	return np.array(x_test)
x_test=prepare_test(root_test)
print(x_test.shape)
print("*******************************************************************************************")
print(x_test)
result=model.predict(x_test)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(result[0])
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
# img=cv2.imread(root+'01-05 10.15.27_2.jpg')
# print(type(img))
# img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# print("++++++++++++")
# cv2.imshow("image",img)
# cv2.waitKey(10000)
# print("++end+++++")
   