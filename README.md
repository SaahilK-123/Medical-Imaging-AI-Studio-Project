## U-Net Implementation with Keras 3.0 and TensorFlow in Python

This project implements the U-Net Architecture using the Keras 3.0 Deep Machine Learning Library with TensorFlow as its backend.

## Full Stack Application

This Medical Imaging AI Project combines a FastAPI Backend with two types of Frontends: A Streamlit-based UI for Image Interaction and a HTML / JavaScript Dashboard for advanced visualisation.
The system simulates a clinically-applicable website where users (both Medical Professionals like Radiologists and even members of the General Public) can anonymously upload medical images to the website and view their results.

## To run the application locally, follow these steps:
1. Create a virtual environment (.venv)

2. Follow all the steps in this setup video [TensorFlow Setup Video](https://www.youtube.com/watch?v=1y8RM4pzM0s) to correctly install the TensorFlow Backend and verify installation.
   
   NOTE 1: The following version will install Tensorflow v2.0 which comes pre-installed with Keras 3.0 hence pip installing Keras will not be necessary.

   NOTE 2: To explicitly check and verify each dependency is installed correctly you can follow the comments placed at the top of the [ Med-AI.py ] file prior to the " AI Scripting Begins Here " comment.

   NOTE 3: To run Keras 3.0 with TensorFlow the latest supported Python version is [ 3.11.0 ]. Set this as the Local version to run the project (alongside any other Python versions that may exist on the device)

4. Define the IMG_DIR variable with the location of the Training Data Folder (it may be necessary to download the Training Data Folder first and then 'Copy as Path')
   NOTE: If you do not wish to use this method then consider configuring a (.env) file with this and other variables that contain sensitive information.

5. Run the [ Med-AI.py ] file and save the resultant model of the file.

6. Run the FastAPI Backend: **uvicorn backend.main:app --reload**

7. Run the Streamlit Frontend: **streamlit run webapp/app.py**

### Credits:

Kst5681 (Link to Profile: https://github.com/Kst5681) for creating the sample website.
