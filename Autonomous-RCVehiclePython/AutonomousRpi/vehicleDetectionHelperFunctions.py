import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from keras.optimizers import Adam
from keras import backend as K

def get_df_vehicles():
    dir_label = ['/home/iftimie/PycharmProjects/object-dataset',
                 '/home/iftimie/PycharmProjects/object-detection-crowdai']
    df_files1 = pd.read_csv(dir_label[1] + '/labels.csv', header=0)
    df_vehicles1 = df_files1[(df_files1['Label'] == 'Car') | (df_files1['Label'] == 'Truck')].reset_index()
    df_vehicles1 = df_vehicles1.drop('index', 1)
    df_vehicles1['File_Path'] = dir_label[1] + '/' + df_vehicles1['Frame']
    df_vehicles1 = df_vehicles1.drop('Preview URL', 1)
    df_files2 = pd.read_csv('/home/iftimie/PycharmProjects/object-dataset/labels.csv', header=None, delimiter=r"\s+", names=list('abcdefgh'))
    df_files2.columns = ['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label', 'RM']
    df_vehicles2 = df_files2[(df_files2['Label'] == 'car') | (df_files2['Label'] == 'truck')].reset_index()
    df_vehicles2 = df_vehicles2.drop('index', 1)
    df_vehicles2 = df_vehicles2.drop('RM', 1)
    df_vehicles2 = df_vehicles2.drop('ind', 1)

    df_vehicles2['File_Path'] = dir_label[0] + '/' + df_vehicles2['Frame']
    ### Combine data frames

    df_vehicles = pd.concat([df_vehicles1, df_vehicles2]).reset_index()
    df_vehicles = df_vehicles.drop('index', 1)
    df_vehicles.columns = ['File_Path', 'Frame', 'Label', 'ymin', 'xmin', 'ymax', 'xmax']
    # print(df_vehicles.head())
    # print(df_vehicles.tail())
    return df_vehicles

def get_df_vehicles_camin():

    df_files2 = pd.read_csv('/home/iftimie/PycharmProjects/object-dataset/labels.csv', header=None, delimiter=r"\s+", names=list('abcdefgh'))
    df_files2.columns = ['Frame', 'xmin', 'xmax', 'ymin', 'ymax', 'ind', 'Label', 'RM']
    df_vehicles2 = df_files2[(df_files2['Label'] == 'car') ].reset_index()
    df_vehicles2 = df_vehicles2.drop('index', 1)
    df_vehicles2 = df_vehicles2.drop('RM', 1)
    df_vehicles2 = df_vehicles2.drop('ind', 1)
    df_vehicles2['File_Path'] ='/home/iftimie/PycharmProjects/object-dataset' + '/' + df_vehicles2['Frame']
    cols = df_vehicles2.columns.tolist()
    cols = [cols[6],cols[0],cols[5],cols[3],cols[1],cols[4],cols[2]]
    df_vehicles2 = df_vehicles2[cols]
    # print (df_vehicles2.head())
    # print (df_vehicles2.tail())
    return df_vehicles2

def augment_brightness_camera_images(image):

    ### Augment brightness
    #image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image[:, :] = image[:, :] * random_bright #image1[:,:,2] = image1[:,:,2]*random_bright
    #image1 = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image #return image1

def trans_image(image,bb_boxes_f,trans_range):
    # Translation augmentation
    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2

    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols,channels = image.shape #rows,cols,channels = image.shape
    bb_boxes_f['xmin'] = bb_boxes_f['xmin']+tr_x
    bb_boxes_f['xmax'] = bb_boxes_f['xmax']+tr_x
    bb_boxes_f['ymin'] = bb_boxes_f['ymin']+tr_y
    bb_boxes_f['ymax'] = bb_boxes_f['ymax']+tr_y

    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr,bb_boxes_f

def stretch_image(img,bb_boxes_f,scale_range):
    # Stretching augmentation

    bb_boxes_f = bb_boxes_f.copy(deep=True)

    tr_x1 = scale_range*np.random.uniform()
    tr_y1 = scale_range*np.random.uniform()
    p1 = (tr_x1,tr_y1)
    tr_x2 = scale_range*np.random.uniform()
    tr_y2 = scale_range*np.random.uniform()
    p2 = (img.shape[1]-tr_x2,tr_y1)

    p3 = (img.shape[1]-tr_x2,img.shape[0]-tr_y2)
    p4 = (tr_x1,img.shape[0]-tr_y2)

    pts1 = np.float32([[p1[0],p1[1]],
                   [p2[0],p2[1]],
                   [p3[0],p3[1]],
                   [p4[0],p4[1]]])
    pts2 = np.float32([[0,0],
                   [img.shape[1],0],
                   [img.shape[1],img.shape[0]],
                   [0,img.shape[0]] ]
                   )

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    img = np.array(img,dtype=np.uint8)

    bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
    bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0])/(p2[0]-p1[0])*img.shape[1]
    bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1])/(p3[1]-p1[1])*img.shape[0]
    bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1])/(p3[1]-p1[1])*img.shape[0]

    return img,bb_boxes_f

def get_image_name(df,ind,size=(640,300),augmentation = False,trans_range = 20,scale_range=20):
    ### Get image by name

    file_name = df['File_Path'][ind]
    img = cv2.imread(file_name)
    img_size = np.shape(img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #before it was img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    name_str = file_name.split('/')
    name_str = name_str[-1]
    #print(name_str)
    #print(file_name)
    bb_boxes = df[df['Frame'] == name_str].reset_index()
    img_size_post = np.shape(img)

    if augmentation == True:
        img,bb_boxes = trans_image(img,bb_boxes,trans_range)
        img,bb_boxes = stretch_image(img,bb_boxes,scale_range)
        img = augment_brightness_camera_images(img)

    bb_boxes['xmin'] = np.round(bb_boxes['xmin']/img_size[1]*img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax']/img_size[1]*img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin']/img_size[0]*img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax']/img_size[0]*img_size_post[0])
    bb_boxes['Area'] = (bb_boxes['xmax']- bb_boxes['xmin'])*(bb_boxes['ymax']- bb_boxes['ymin'])
    #bb_boxes = bb_boxes[bb_boxes['Area']>400]


    return name_str,img,bb_boxes

def get_mask_seg(img,bb_boxes_f):

    #### Get mask

    img_mask = np.zeros_like(img[:, :,0]) #img_mask = np.zeros_like(img[:,:,0])
    for i in range(len(bb_boxes_f)):
        #plot_bbox(bb_boxes,i,'g')
        bb_box_i = [bb_boxes_f.iloc[i]['xmin'],bb_boxes_f.iloc[i]['ymin'],
                bb_boxes_f.iloc[i]['xmax'],bb_boxes_f.iloc[i]['ymax']]
        img_mask[int(bb_box_i[1]):int(bb_box_i[3]),int(bb_box_i[0]):int(bb_box_i[2])        ]= 1.
        img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))
    return img_mask

def plot_im_mask(im,im_mask):
    ### Function to plot image mask

    im = np.array(im,dtype=np.uint8)
    im_mask = np.array(im_mask,dtype=np.uint8)
    plt.subplot(1,3,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(im_mask[:,:,0])
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(cv2.bitwise_and(im,im,mask=im_mask));
    plt.axis('off')
    plt.show();

def plot_bbox(bb_boxes,ind_bb,color='r',linewidth=2):
    ### Plot bounding box

    bb_box_i = [bb_boxes.iloc[ind_bb]['xmin'],
                bb_boxes.iloc[ind_bb]['ymin'],
                bb_boxes.iloc[ind_bb]['xmax'],
                bb_boxes.iloc[ind_bb]['ymax']]
    plt.plot([bb_box_i[0],bb_box_i[2],bb_box_i[2],
                  bb_box_i[0],bb_box_i[0]],
             [bb_box_i[1],bb_box_i[1],bb_box_i[3],
                  bb_box_i[3],bb_box_i[1]],
             color,linewidth=linewidth)

def plot_im_bbox(im,bb_boxes):
    ### Plot image and bounding box
    plt.imshow(im)
    for i in range(len(bb_boxes)):
        plot_bbox(bb_boxes,i,'g')

        bb_box_i = [bb_boxes.iloc[i]['xmin'],bb_boxes.iloc[i]['ymin'],
                bb_boxes.iloc[i]['xmax'],bb_boxes.iloc[i]['ymax']]
        plt.plot(bb_box_i[0],bb_box_i[1],'rs')
        plt.plot(bb_box_i[2],bb_box_i[3],'bs')
    plt.axis('off');

def generate_train_batch(df_vehicles,batch_size = 32,img_rows=480,img_cols=640):
    batch_images = np.zeros((batch_size, img_rows, img_cols,3)) #batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    print("len is",len(df_vehicles))
    while 1:
        for i_batch in range(batch_size):

            #i_line = np.random.randint(len(df_vehicles)-2000)
            i_line = np.random.randint(len(df_vehicles))
            name_str,img,bb_boxes = get_image_name(df_vehicles,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=True,
                                                   trans_range=50,
                                                   scale_range=50
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        if ((np.max(nonzeroy)-np.min(nonzeroy)>50) & (np.max(nonzerox)-np.min(nonzerox)>50)):
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255),6)
    # Return the image
    return img

def test_new_img(model,img,new_cols,new_rows):
    img = cv2.resize(img,(new_cols, new_rows))
    img = np.reshape(img,(1,new_rows, new_cols,3))
    pred = model.predict(img)
    return pred,img[0]

def get_BB_new_img(model,img,img_cols,img_rows):
    # Take in RGB image
    pred,img = test_new_img(model=model,img=img,new_cols=img_cols,new_rows=img_rows)
    img  = np.array(img,dtype= np.uint8)
    img_pred = np.array(255*pred[0],dtype=np.uint8)
    heatmap = img_pred[:,:,0]
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

smooth = 1

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

def train_and_save(model,df_vehicles):
    training_gen = generate_train_batch(df_vehicles=df_vehicles, batch_size=6, img_rows=480, img_cols=640)
    model.compile(optimizer=Adam(lr=1e-4), loss=IOU_calc_loss, metrics=[IOU_calc])
    history = model.fit_generator(training_gen, samples_per_epoch=2000, nb_epoch=10)
    model.save('model_detect_SmallUnet.h5')

def test_image(model,img_cols,img_rows,test_img):
    im = cv2.imread(test_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



    im = cv2.imread(test_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pred, im = test_new_img(model, im, new_cols=img_cols, new_rows=img_rows)
    im = np.array(im, dtype=np.uint8)
    im_pred = np.array(255 * pred[0], dtype=np.uint8)
    rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]

    img_pred = cv2.addWeighted(rgb_mask_pred, 0.55, im, 1, 0)

    draw_img = get_BB_new_img(model=model, img=im, img_cols=img_cols, img_rows=img_rows)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_pred)
    plt.title('Segmentation')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(draw_img)
    plt.title('Bounding Box')
    plt.axis('off');
    plt.show()

def getCenterPointsAndDraw(imageToDraw,binaryImage):
    heatmap = binaryImage[:, :, 0]
    labels = label(heatmap)
    centerPoints = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        if ((np.max(nonzeroy) - np.min(nonzeroy) > 50) & (np.max(nonzerox) - np.min(nonzerox) > 50)):
            x1= np.min(nonzerox)
            y1= np.min(nonzeroy)
            x2 = np.max(nonzerox)
            y2 = np.max(nonzeroy)
            bbox = ((x1, y1), (x2, y2))
            centerPoints.append((int((x1+x2)/2),int((y1+y2)/2)))
            # Draw the box on the image
            cv2.rectangle(imageToDraw, bbox[0], bbox[1], (255,25,255), 6)
    # Return the image
    return imageToDraw,centerPoints

def getCenterPoints(binaryImage):
    heatmap = binaryImage[:, :, 0]
    labels = label(heatmap)
    centerPoints = []
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        x1 = np.min(nonzerox)
        y1 = np.min(nonzeroy)
        x2 = np.max(nonzerox)
        y2 = np.max(nonzeroy)
        if ((y2 - y1) > 50 & (x2 - x1) > 50):
            bbox = ((x1, y1), (x2, y2))
            centerPoints.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
    return centerPoints