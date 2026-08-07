    #1.SETUP env
env will be including KAGGLE_API_TOKEN which is created using your kaggle account

    #2.SETUP DATASET
run extract_data_kaggle.py for downloading and extracting the dataset from kaggle to your local

    #3.PROCESSING IMAGES FROM DATASET
        ##3.1.PROCESSING IMAGES USING NOTEBOOKS
run 01_basic_processing.ipynb and 02_enhancement.ipynb for processing images from downloaded dataset.
- 1st notebook has built-in functions to handle rotate/flip/crop/denoise/edge/background_subtract
- 2nd notebook has built-in functions to handle hist_eq/clahe_enhance/gamma_correction/unsharp_mask

        ##3.2.INTERFACE APP USING STREAMLIT
streamlit run .\app.py