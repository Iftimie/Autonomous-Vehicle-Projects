from networkFunctions import *
from vehicleDetectionHelperFunctions import *

img_rows = 480
img_cols = 640

model = get_small_unet(img_rows=img_rows,img_cols=img_cols)

model.load_weights("model_detect_SmallUnet.h5")
#df_vehicles = get_df_vehicles()
#df_vehicles = get_df_vehicles_camin()
#train_and_save(model=model,df_vehicles=df_vehicles)

model.load_weights("model_detect_SmallUnet.h5")

test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/img691.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/1478896589590329220.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/1478896601017813080.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/1478897609342522869.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/img263.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/img299.jpg')
test_image(model=model,img_cols=img_cols,img_rows=img_rows,test_img='/home/iftimie/PycharmProjects/object-dataset/img324.jpg')








