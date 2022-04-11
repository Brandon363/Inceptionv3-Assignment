import streamlit as st
pip install opencv-python
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob
import shutil


#------------------------------------Page 1 ----------------------------------------
rad = st.sidebar.radio("Navigation", ["Home", "About The Developers"])

if rad == "Home":
    #################################selecting video ################################
    st.title("Inception V3 Model")


    predicted_object = st.text_input("Enter the name of the object to be searched eg sports_car")

    predict = predicted_object

    picked_video = st.file_uploader("Select the Video To Be Analyzed")



    if picked_video:
        if st.checkbox("Show Selected Video"):
            st.video(picked_video)

    if st.button("Search"):
        st.success("Searching started")


        #######################################################################################    

        ################################# saving video to local folder #######################

        def save_uploaded_file(uploaded_file):

            try:

                with open(os.path.join('static/videos',uploaded_file.name),'wb') as f:

                    f.write(uploaded_file.getbuffer())

                return 1    

            except:

                return 0

        #######################################################################################

        ###################################### chopping video ################################


        #uploaded_file = st.file_uploader("Upload Video")

        if not os.path.exists('static'):
                    os.makedirs('static')

        if not os.path.exists('static/pictures'):
                    os.makedirs('static/pictures')

        if not os.path.exists('static/pictures/selected'):
                    os.makedirs('static/pictures/selected') 

        if picked_video is not None:

            if save_uploaded_file(picked_video):

                vid            = cv2.VideoCapture(os.path.join('static/videos',picked_video.name))
                
                currentframe   = 0
                
                    
                while (True):
                    success, frame = vid.read()
                    if success:
                        #cv2.imshow('output', frame)
                        cv2.imwrite(f'static/pictures/frame' + str(currentframe) + '.jpg', frame)


                        currentframe +=1
                    if not success:
                        break
                    
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                vid.release()
                cv2.destroyAllWindows()

        #######################################################################################  



        ##############################################################################################

        #load the model
        #model_inception = load_model('model/inception.h5')
        model_inception = InceptionV3()


        #load an image from file


        for filepath in glob.iglob('static/pictures/*.jpg'):

            real_image = load_img(filepath, target_size=(299, 299))
                    # convert the image pixels to a numpy array
            image = img_to_array(real_image)
                    # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    # prepare the image for the Inception model
            image = preprocess_input(image)
                    # predict the probability across all output classes
            yhat = model_inception.predict(image)
                    # convert the probabilities to class labels
            label = decode_predictions(yhat)
                    # retrieve the most likely result, e.g. highest probability
            label = label[0][0]
            
                    # print the classification
                    #print('%s (%.2f%%)' % (label[1], label[2]*100))

            pred_count = 1

            if label[1] == predict:        
                src = "%s" % (filepath)
                dst = "static/pictures/selected"
                shutil.copy2(src, dst)
            else:
                    #print("no specified object")
                pass



        #########################################################################################################

        ############################### directories ##############################################################\

        image_path = "static/pictures/selected/*.jpg"
        images_folder = "static/pictures/selected/"
        images_sel    = os.listdir(images_folder)
        selected_images = [cv2.imread(selected_image) for selected_image in glob.glob(image_path)]
        #glob.glob(image_path)

        ############################################################################################################


        #################################################  code for pic plots  ###########################################


        for filepath2 in glob.iglob('static/pictures/selected/*.jpg'):
            for i in range(len(images_sel)):

                st.image(filepath2, caption = images_sel[i], width = 500)




        st.success("Searching done")
        st.balloons()
        
elif rad == "About The Developers":
    col1, col2 = st.columns(2)


    with col1:
        st.header("Brandon Mutombwa")
        bran_pic = load_img("static/brandon_pic.jpg", target_size=(300, 350))
        st.image(bran_pic, caption = "R204739S")
        
    with col2:
        st.header("Tafadzwa Nyandoro")
        taf_pic = load_img("static/tafadzwa_pic.jpg", target_size=(300, 350))
        st.image(taf_pic, caption = "R205761T")

    st.write("The above are two developers aspiring data scientists who, by the time this app was deployed, where completing their second accademic year at the University of Zimbabwe")



    
