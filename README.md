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

## Citation

If you find this repository useful, please cite our paper: 
```
@article{mu2025drbp,
  title={DRBP-EDP: classification of DNA-binding proteins and RNA-binding proteins using ESM-2 and dual-path neural network},
  author={Mu, Qiang and Yu, Guoping and Zhou, Guomin and He, Yubing and Zhang, Jianhua},
  journal={NAR Genomics and Bioinformatics},
  volume={7},
  number={2},
  pages={lqaf058},
  year={2025},
  publisher={Oxford University Press}
}
```

## Other methods
| Methods | Prediction Types | Server Links | Citations  |
|--------------------------|-----------|----------|------------------------------|
| **PlDBPred** | DBPs | [https://iasri-sg.icar.gov.in/pldbpred/](https://iasri-sg.icar.gov.in/pldbpred/) | [PlDBPred: a novel computational model for discovery of DNA binding proteins in plants](https://doi.org/10.1093/bib/bbac483) |
| **DPP-PseAAC** | DBPs | [http://77.68.43.135:8080/DPP-PseAAC/](http://77.68.43.135:8080/DPP-PseAAC/) | [DPP-PseAAC: A DNA-binding protein prediction model using Chou’s general PseAAC](https://doi.org/10.1016/j.jtbi.2018.05.006) |
| **ProkDBP** | DBPs | [https://iasri-sg.icar.gov.in/prokdbp/](https://iasri-sg.icar.gov.in/prokdbp/) | [ProkDBP: Toward more precise identification of prokaryotic DNA binding proteins](https://doi.org/10.1002/pro.5015) |
| **Deep-RBPPred** | RBPs | [http://www.rnabinding.com/Deep_RBPPred/Deep-RBPPred.html](http://www.rnabinding.com/Deep_RBPPred/Deep-RBPPred.html) | [Deep-RBPPred: Predicting RNA binding proteins in the proteome scale based on deep learning](https://doi.org/10.1038/s41598-018-33654-x) |
| **catRAPID signature** | RBPs | [http://s.tartaglialab.com/new_submission/signature](http://s.tartaglialab.com/new_submission/signature) | [catRAPID signature: identification of ribonucleoproteins and RNA-binding regions](https://doi.org/10.1093/bioinformatics/btv629) |
| **RBPLight** | RBPs | [https://iasri-sg.icar.gov.in/rbplight/](https://iasri-sg.icar.gov.in/rbplight/) | [RBPLight: a computational tool for discovery of plant-specific RNA-binding proteins using light gradient boosting machine and ensemble of evolutionary features](https://doi.org/10.1093/bfgp/elad016) |
| **DeepDRBP-2L** | DBPs/RBPs | [http://bliulab.net/DeepDRBP-2L](http://bliulab.net/DeepDRBP-2L) | [DeepDRBP-2L: A New Genome Annotation Predictor for Identifying DNA-Binding Proteins and RNA-Binding Proteins Using Convolutional Neural Network and Long Short-Term Memory](https://doi.org/10.1109/TCBB.2019.2952338) |
| **iDRBP-ECHF** | DBPs/RBPs | [http://bliulab.net/iDRBP-ECHF](http://bliulab.net/iDRBP-ECHF) | [iDRBP-ECHF: Identifying DNA- and RNA-binding proteins based on extensible cubic hybrid framework](https://doi.org/10.1016/j.compbiomed.2022.105940) |
| **iDRBP_MMC** | DBPs/RBPs | [http://bliulab.net/iDRBP_MMC](http://bliulab.net/iDRBP_MMC) | [iDRBP_MMC: Identifying DNA-Binding Proteins and RNA-Binding Proteins Based on Multi-Label Learning Model and Motif-Based Convolutional Neural Network](https://doi.org/10.1016/j.jmb.2020.09.008) |
| **iDRBP-EL** | DBPs/RBPs | [http://bliulab.net/iDRBP-EL](http://bliulab.net/iDRBP-EL) | [iDRBP-EL: Identifying DNA- and RNA- Binding Proteins Based on Hierarchical Ensemble Learning](https://doi.org/10.1109/TCBB.2021.3136905) |
| **IDRBP-PPCT** | DBPs/RBPs | [http://bliulab.net/IDRBP-PPCT](http://bliulab.net/IDRBP-PPCT) | [IDRBP-PPCT: Identifying Nucleic Acid-Binding Proteins Based on Position-Specific Score Matrix and Position-Specific Frequency Matrix Cross Transformation](https://doi.org/10.1109/TCBB.2021.3069263) |
| **iDRPro-SC** | DBPs/RBPs | [http://bliulab.net/iDRPro-SC](http://bliulab.net/iDRPro-SC) | [iDRPro-SC: identifying DNA-binding proteins and RNA-binding proteins based on subfunction classifiers](https://doi.org/10.1093/bib/bbad251) |
