# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches

# https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/
# https://github.com/Shenggan/BCCD_Dataset/blob/master/BCCD/JPEGImages/BloodImage_00398.jpg

# read the csv file using read_csv function of pandas
train = pd.read_csv('train.csv',delimiter=";")
train.head()

# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

# reading single image using imread function of matplotlib
image = plt.imread('images/BloodImage_00398.jpg')
plt.imshow(image)

# Number of unique training images
train['image_names'].nunique()

# Number of classes
train['cell_type'].value_counts()


fig = plt.figure()

#add axes to the image
ax = fig.add_axes([0,0,1,1])

# read and plot the image
#image = plt.imread('images/BloodImage_00398.jpg')
#plt.imshow(image)

# iterating over the image for different objects
for _,row in train[train.image_names == "BloodImage_00398.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    if row.cell_type == 'RBC':
        color = (255, 0, 0)  
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        color = (125, 0, 0) 
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        color = (240, 10, 0) 
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    ax.add_patch(rect)
    
    # Draw a rectangle with blue line borders of thickness of 2 px 
    image = cv2.rectangle(image, (xmin,xmax), (ymin, ymax), color, thickness) 
    
    

plt.imshow(image)






data = pd.DataFrame()
data['format'] = train['image_names']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['cell_type'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')