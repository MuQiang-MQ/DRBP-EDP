**DRBP-EDP** has been developed in both executable and web-based versions.

- **The executable version** of DRBP-EDP can be accessed and downloaded from [Hugging Face](https://huggingface.co/MuQiang/DRBP_EDP).

- **The web-based version** involves the following steps to set up and run:
    1. **Clone the Repository**: First, clone the DRBP-EDP repository from GitHub to your local machine:

        `git clone https://github.com/MuQiang-MQ/DRBP-EDP.git`
  
    2. **Install Conda (if not already installed)**: If you don't have Conda installed, download and install Anaconda or Miniconda.
 
    3. **Create a Conda Environment**: Create a new Conda environment for the project. You can specify the Python version you want to use (preferably Python 3.9 or higher) with the following command:

       `conda create --name drbp-edp-env python=3.9`

    4. **Activate the Conda Environment**: After creating the environment, activate it with:

       `conda activate drbp-edp-env`

    5. **Install Dependencies**: Ensure that you are in the root directory of the project (where `requirements.txt` is located). Use the following command to navigate to that directory: 
 
       `cd /d path/to/your/project`
 
       Then, install all required dependencies by running:
        ```
       # Before running the next command, ensure the environment is activated
       pip install -r requirements.txt
    6. **Download the Model Files**: The web-based version of DRBP-EDP requires the model checkpoint files to function properly. Please visit the Hugging Face page for the model and download the model folder from [this link](https://huggingface.co/MuQiang/DRBP_EDP).
 
       The model folder contains:
        - `esm2_t33_650M_UR50D` (the model checkpoint file)
        - `best_model_stage1.pth`
        - `best_model_stage2.pth`
      
       After downloading, place these files into the DRBP-EDP folder.

    7. **Verify the Directory Structure**: Ensure that the directory structure looks like the following:

         ```
         DRBP-EDP/
          ├── drbp_edp_web_local.py
          ├── requirements.txt
          ├── icon.ico
          ├── model/                  <-- This should contain the downloaded model files
          │   ├── esm2_t33_650M_UR50D
          │   ├── best_model_stage1.pth
          │   ├── best_model_stage2.pth
          ├── ... (other files and directories)
    8. **Run the Web-Based Application**: Now you are ready to run the web application. In the terminal, from the project directory, execute:
 
        `streamlit run drbp_edp_web_local.py`

       This will start a local server, and the terminal should show output similar to:

       ```
       You can now view your Streamlit app in your browser.
       Local URL:  http://localhost:8501
       Network URL:  http://<your-network-ip>:8501
    9. **Access the Application**: Once the Streamlit app is running, it should automatically open in your browser. If it doesn't, or if you want to manually check, simply open your browser and navigate to `http://localhost:8501` to view and interact with the web application.
