# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:39:32 2018

@author: Krish.Naik
"""

%%writefile score.py
# ***********
# Assumptions
# ***********
# Unstructured classification code is applicable to textual feature
# Target column must be categorical
from keras.layers import Input, Lambda, Dense, Flatten
import traceback
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import base64
from io import BytesIO
import json
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.datastore import Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from sklearn.externals import joblib

#Intialization Function
def init():
    global model
    # retreive the path to the model file using the model name
    model_path1 = Model.get_model_path('facekrish')
    model = load_model(model_path1)
    
def run(raw_data):
    
    encoded_image = np.array(json.loads(raw_data)['encimage'])
    try:
        
        decoded_image = base64.b64decode(str(encoded_image))
        dimensions=(100,100)
        img = Image.open(BytesIO(decoded_image))
        img = img.convert('RGB')
        img = img.resize(dimensions, Image.ANTIALIAS)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        preds = model.predict(img_data)
        dark_circles=float(model.predict(img_data)[0][0])
        kajal=float(model.predict(img_data)[0][1])
        lipstick=int(model.predict(img_data)[0][2])
        long_hair=float(model.predict(img_data)[0][3])
        short_hair=float(model.predict(img_data)[0][4])
        list_products_darkcircles=["Pond's Triple Vitamin Moisturising Lotion","Pond's Moisturising Cold Cream","Pond's Gold Radiance Serum","Pond's Sandal Radiance Natural Sunscreen Talc"]
        list_products_longhair=['Tresemme Keratin Shampoo','Sunsilk Shine Shampoo','Dove Shampoo']
        list_products_kajal=['Lakme Eyeliner','Lakme Kajal']
        list_products_shorthair=['Tresemme','Sunsilk Black']
        list_products_lipstick=['Lakme 9 to 5 Naturale Matte Sticks Lipstick','Lakme 9 to 5 Primer'] 
        lst=[]

        prod_dict={}

        if dark_circles>0:  
    
            random_choice=np.random.choice(list_products_darkcircles)
            lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/71QQTSc88WL._SL1500_.jpg","product_url":"https://www.amazon.in/Ponds-Age-Miracle-Massager-Kit/dp/B07BNQ8XGS/ref=sr_1_1_sspa?ie=UTF8&qid=1542876771&sr=8-1-spons&keywords=ponds&psc=1"})

                
        if kajal>0.7:
            random_choice=np.random.choice(list_products_kajal)
            lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/51FaZZnFl1L._SL1000_.jpg","product_url":"https://www.amazon.in/Lakme-Insta-Eye-Liner-Black/dp/B006LXBSYM/ref=sr_1_2?s=beauty&ie=UTF8&qid=1542876914&sr=1-2&keywords=Lakme+Eyeliner"})
    
        if lipstick>0.7:
            random_choice=random.choice(list_products_lipstick)
            lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/51NwzGVujLL._SL1000_.jpg","product_url":"https://www.amazon.in/Lakme-Primer-Matte-Color-Berry/dp/B010BCYMDC/ref=sr_1_1?ie=UTF8&qid=1542876954&sr=8-1&keywords=Lakme+9+to+5+Primer"})
       
        if (long_hair>0 or short_hair>0) :
            random_choice=np.random.choice(list_products_longhair)
            if(random_choice.lower().find("sunsilk")!= -1):
              #  prod_dict["products"]={"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/616l26X4I0L._SL1000_.jpg","product_url":"https://www.amazon.in/Sunsilk-Stunning-Black-Shine-Shampoo/dp/B01MXKHGGK/ref=sr_1_1?ie=UTF8&qid=1542884871&sr=1-1&keywords=Sunsilk+shine"}
                lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/616l26X4I0L._SL1000_.jpg","product_url":"https://www.amazon.in/Sunsilk-Stunning-Black-Shine-Shampoo/dp/B01MXKHGGK/ref=sr_1_1?ie=UTF8&qid=1542884871&sr=1-1&keywords=Sunsilk+shine"})
            elif(random_choice.lower().find("dove")!= -1):
                #prod_dict["products"]={"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/61XcibvMhJL._SL1000_.jpg","product_url":"https://www.amazon.in/Dove-Daily-Shine-Shampoo-650ml/dp/B01G3K83AE/ref=sr_1_2?srs=9574332031&ie=UTF8&qid=1542884963&sr=8-2&keywords=dove+shamppoo"}
                lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/61XcibvMhJL._SL1000_.jpg","product_url":"https://www.amazon.in/Dove-Daily-Shine-Shampoo-650ml/dp/B01G3K83AE/ref=sr_1_2?srs=9574332031&ie=UTF8&qid=1542884963&sr=8-2&keywords=dove+shamppoo"})
            else:
                #prod_dict["products"]={}
               # prod_dict["products"]={"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/610-uFXNdOL._SL1000_.jpg","product_url":"https://www.amazon.in/TRESemme-Spa-Rejuvenation-Shampoo-580ml/dp/B07F7LRVGY/ref=sr_1_3?srs=9574332031&ie=UTF8&qid=1542885085&sr=8-3&keywords=tresseme+keratin+shampoo"}
                lst.append({"products_name":random_choice,"image_url":"https://images-na.ssl-images-amazon.com/images/I/610-uFXNdOL._SL1000_.jpg","product_url":"https://www.amazon.in/TRESemme-Spa-Rejuvenation-Shampoo-580ml/dp/B07F7LRVGY/ref=sr_1_3?srs=9574332031&ie=UTF8&qid=1542885085&sr=8-3&keywords=tresseme+keratin+shampoo"})
     
                
        lst={"Products":lst}
        return json.dumps(lst)
                                         
    except Exception as e:
        traceback.print_exc()
        er_details=str(e)
        print ("Execution terminated")
        result={
                "Status": "Failure",
                "Error Message": er_details
            }
        return json.dumps(result)