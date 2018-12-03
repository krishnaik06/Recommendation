import cv2
import numpy as np
from PIL import Image
from keras import models
import random

#Load the saved model
model = models.load_model('facefeatures_new_model.h5')


def predict_classifier():

    while True:
            _, frame = video.read()
            #Convert the captured frame into RGB
            im = Image.fromarray(frame, 'RGB')
    
            #Resizing into 128x128 because we trained the model with this image size.
            im = im.resize((100,100))
            img_array = np.array(im)
            #Our keras model used a 4D tensor, (images x height x width x channel)
            #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
    
            #Calling the predict method on model to predict 'me' on the image
            prediction =model.predict(img_array)
            
            dark_circles=int(model.predict(img_array)[0][0])
            kajal=int(model.predict(img_array)[0][1])
            lipstick=int(model.predict(img_array)[0][2])
            long_hair=int(model.predict(img_array)[0][3])
            short_hair=int(model.predict(img_array)[0][4])
            list_products_darkcircles=['Lux','Ponds','Rexona','Dove soap']
            list_products_longhair=['Pantene','Tresemme','Sunsilk','Dove Sahampoo']
            list_products_kajal=['Lakme eyeliner','Lakme kajal']
            list_products_shorthair=['Pantene shine','Tresemme','Sunsilk black']
            list_products_lipstick=['Lakme 9 to 5 Naturale Matte Sticks Lipstick','Lakme 9 to 5 Primer']
            lst=[]
            print(prediction)
            if dark_circles>0:
                random_choice=random.choice(list_products_darkcircles)
                icon_img = cv2.imread("download.jpg")
                icon_img1= cv2.resize(icon_img, (100,100))
                x_offset=y_offset=5
                frame[y_offset:y_offset+icon_img1.shape[0], x_offset:x_offset+icon_img1.shape[1]] = icon_img1
                cv2.putText(frame,"Ponds Cold Cream",(x_offset+icon_img1.shape[1], y_offset+icon_img1.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                print("You have Dark circles ")
                print("Product recommended is {}".format(random_choice))
            elif long_hair>0.5:
                random_choice=random.choice(list_products_longhair)
                icon_img = cv2.imread("sunsilk.jpg")
                icon_img1= cv2.resize(icon_img, (100,100))
                x_offset=y_offset=5
                frame[y_offset:y_offset+icon_img1.shape[0], x_offset:x_offset+icon_img1.shape[1]] = icon_img1
                cv2.putText(frame,"Sunsilk Shine",(x_offset+icon_img1.shape[1], y_offset+icon_img1.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                print("You have long hair ")
                print("Product recommended is {}".format(random_choice))
            else:
                random_choice=random.choice(list_products_shorthair)
                icon_img = cv2.imread("Tresseme.jpg")
                icon_img1= cv2.resize(icon_img, (100,100))
                x_offset=y_offset=5
                frame[y_offset:y_offset+icon_img1.shape[0], x_offset:x_offset+icon_img1.shape[1]] = icon_img1
                cv2.putText(frame,"Tresemme",(x_offset+icon_img1.shape[1], y_offset+icon_img1.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                print("You have short hair ")
                print("Product recommended is {}".format(random_choice))
            
            #GetProductRecommendation(prediction)
            
            cv2.imshow('Video', frame)
            key=cv2.waitKey(1)
            if key == ord('q'):
                    break
    video.release()
    cv2.destroyAllWindows()
    
    
video = cv2.VideoCapture(0)


        
    
predict_classifier()

    
