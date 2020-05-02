# importing required libraries
# https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/

import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches

# el orden es name, bullet (type), x1, x2, y1, y2

# read the csv file using read_csv function of pandas
train = pd.read_csv('train.csv',delimiter=";")
train.head()

# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

# reading single image using imread function of matplotlib
image = plt.imread('train_images/IMG_6974.JPG')
plt.imshow(image)

# Number of unique training images
train['image_names'].nunique()

# Number of classes
train['type'].value_counts()


#fig = plt.figure()

#add axes to the image
#ax = fig.add_axes([0,0,1,1])


# iterating over the image for different objects
for _,row in train[train.image_names == "IMG_6974.JPG"].iterrows():
    xmin = int(row.xmin)
    xmax = int(row.xmax)
    ymin = int(row.ymin)
    ymax = int(row.ymax)
    
    width = xmax - xmin
    height = ymax - ymin
    
    print("xmin:"+str(xmin))
    print("xmax:"+str(xmax))
    print("ymin:"+str(ymin))
    print("ymax:"+str(ymax))
    
    # assign different color to different classes of objects
    color = (255, 0, 0)  
    edgecolor = 'r'
    #ax.annotate('bullet', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    #rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    #ax.add_patch(rect)
    
    # Draw a rectangle with blue line borders of thickness of 2 px 
    image = cv2.rectangle(image, (xmin,ymin), (xmax, ymax), color, thickness)   
    
    cv2.imwrite("detected.jpg", image)
#plt.imshow(image)



data = pd.DataFrame()
data['format'] = train['image_names']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(int(train['xmin'][i])) + ',' + str(int(train['ymin'][i])) + ',' + str(int(train['xmax'][i])) +    ',' + str(int(train['ymax'][i])) + ',' + train['type'][i] 

data.to_csv('annotate.txt', header=None, index=None, sep=' ')
