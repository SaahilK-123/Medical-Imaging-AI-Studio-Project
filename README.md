## U-Net Implementation with Keras 3.0 and TensorFlow in Python

This project implements the U-Net Architecture using the Keras 3.0 Deep Machine Learning Library with TensorFlow as its backend. Its focus is on medical image segmentation and integrates this model into a Full Stack AI-Powered Application for clinical-style use. Users - such as Radiologists and the General Public - will be able to anonymously upload medical scans and receive processed results. This project simulates a privacy-aware and clinically-applicable workflow.

## Full Stack Application

This Medical Imaging AI Project combines a FastAPI Backend with two types of Frontends: 

1. A Streamlit-based UI for viewing Image Segmented results interactively

2. A HTML / JavaScript Dashboard for advanced visualisation with scope to expand into role-based access levels (different functions for general public user vs 
   medical professional)

## To run the application locally, follow these steps:
1. Create a virtual environment (.venv)

2. Follow all the steps in this setup video [TensorFlow Setup Video](https://www.youtube.com/watch?v=1y8RM4pzM0s) to correctly install the TensorFlow Backend and verify installation.
   
    ⚠️ NOTE 1: The following version will install Tensorflow v2.0 which comes pre-installed with Keras 3.0 hence pip installing Keras will not be necessary.

    ⚠️ NOTE 2: To explicitly check and verify each dependency is installed correctly you can follow the comments placed at the top of the [ Med-AI.py ] file prior 
    to the " AI Scripting Begins Here " comment.

    ⚠️ NOTE 3: To run Keras 3.0 with TensorFlow the latest supported Python version is [ 3.11.0 ]. Set this as the Local version to run the project (alongside any 
    other Python versions that may exist on the device)

4. Define the IMG_DIR variable with the location of the Training Data Folder
   - Option 1: Use Python's 'Copy as Path' to insert the local directory.
   - Option 2: Configure a (.env) file in the main workspace (do not put it inside of a folder or other sub-section) [Recommended for larger projects or cloud-based projects].

6. Run the [ Med-AI.py ] file and save the resultant model of the file.

7. Run the FastAPI Backend: **uvicorn backend.main:app --reload**

8. Run the Streamlit Frontend: **streamlit run webapp/app.py**

### Credits:

[Kst5681](https://github.com/Kst5681) for creating the sample website.
