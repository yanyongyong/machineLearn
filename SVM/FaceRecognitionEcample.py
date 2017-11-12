from __future__ import print_function

from time import time #时间
import logging #log信息
import matplotlib.pylab as plt #绘图工具

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC #向量机


#利用SVM进行人脸识别
print(__doc__)

#打印进展信息
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

#dOwnload the data, if not already on disk and load it as numpy arrays(下载数据集),具体看fetch_lfw_people方法描述
lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)

# introspect the images arrays to find the shapes(通过lfw_people提取shapes)
n_samples, h, w = lfw_people.images.shape
# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data #得到特征向量矩阵
n_features = X.shape[1] #得到的维度（shape[1]列数）

#the label to predict is the id of the person
y = lfw_people.target#对应的不用的人（特征）
target_names = lfw_people.target_names #返回得到人的姓名
n_classes = target_names.shape[0] #返回类（人）数

print("n_shamples: ",n_samples)
print('n_features: ',n_features)
print('n_classes: ',n_classes)

#split into a training and testing set
#X_train 矩阵，y_train 是X_train里面的向量
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#降维度(利用PCA降维度)
n_components = 150 #组成元素的数量，设置为150
print('Extraction the top %d eiaenfaces from %d faces' % (n_components,X_train.shape[0]))
t0 = time() #打印出每一步花的时间
pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X_train) #高纬降为低纬
print('done in %0.3fs' %(time() - t0))

eigenfaces = pca.components_.reshape(n_components,h,w) # 提取一些特征点

print('projectiong the input data on the eigenfaces arthonormal basis')
t0 = time()
X_train_pca = pca.transform(X_train) #矩阵向低纬转换
X_test_pca = pca.transform(X_test)
print('done in %0.3fs ' %(time()-t0))

##################################################################
#Train a SVM classification model(根据降维后的矩阵建立模型)
print('Fitting the classifier to the training set')
t0 = time()
#搜索30对组合（C和gamma）那一对的组合精确度最高
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid) #因为处理图像，SO选择rbf作为核函数,class_weight权重
clf = clf.fit(X_train_pca,y_train) #根据SVM进行建模,保存到clf中
print('done in %0.3fs' %(time() - t0))
print('Best estimator found by grid search:')
print(clf.best_estimator_)

#################################################################
#Quantitative evaluation of the model quality onthe test set

print('Prediction people`s names on the test set')
t0 = time()
y_pred = clf.predict(X_test_pca)
print('done in %0.3fs'%(time()-t0))

print(classification_report(y_test,y_pred,target_names=target_names))#测试标签和真实标签作比较
print(confusion_matrix(y_test,y_pred,labels=range(n_classes))) #N*N的矩阵，比较预测和实际的标签

####################################################################
#Qualitative evaluation of the predictions using eatplotlib
#预测的结果以一种合理的方式表示出来，可视化
def plot_gallery(images,titles,h,w,n_row=3,n_col=4):#images传入的图像
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=0.99,top=.90,hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row,n_col,i + 1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.yticks(())
        plt.yticks(())

#预测的函数归类标签和实际函数标签对应的名字
def title(y_pred,y_test,target_name,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    return 'predicted :%s\ntrue:  %s' %(pred_name,true_name)

prediction_titles = [title(y_pred,y_test,target_names,i) for i in range(y_pred.shape[0])]
plot_gallery(X_test,prediction_titles,h,w)

#plot the gallery of the most significative eignefaces
eigenface_titles = ['eigenface %d' % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenface_titles,h,w)
plt.show()

