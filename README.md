# Diamond Price Prediction

Welcome to the Diamond Price Prediction project! This project is implemented using Python modular programming and aims to predict the price of diamonds based on various features.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Explanation](#project-explanation)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
The Diamond Price Prediction project uses machine learning techniques to predict the price of diamonds. The dataset contains various features that influence the price of diamonds. The project is structured in a modular way, making it easy to understand, maintain, and extend.

## Project Structure
The repository contains the following folders and files:
1. **artifacts**: Stores `model.pkl`, `preprocessor.pkl`, `raw.csv`, `test.csv`, and `train.csv`.
2. **logs**: Stores all runtime logs.
3. **notebooks**: Contains Jupyter notebooks for exploratory data analysis (EDA) and model training:
   - `EDA.ipynb`: Exploratory Data Analysis.
   - `model_training.ipynb`: Model Training.
   - `data/gemstone.csv`: The actual dataset.
4. **src**: Contains the core code organized into components and pipelines:
   - **components**: Includes `data_ingestion.py`, `data_transformation.py`, `model_trainer.py`.
   - **pipelines**: Contains `prediction_pipeline.py` and `training_pipeline.py`.
   - Also includes utility and helper files like `exception.py`, `logger.py`, and `utils.py`.
5. **templates**: Contains the HTML files.
6. **app.py**: The main application file.
7. **README.md**: This file.
8. **requirements.txt**: Lists all the dependencies.
9. **setup.py**: Script for setting up the project.

## Installation
To get a copy of this project up and running on your local machine, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SHP50c/DiamondPricePrediction.git
    cd diamond-price-prediction
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    **Command Prompt:**
    ```cmd
    python -m venv env
    env\Scripts\activate
    ```

3. **Install ipykernel for Jupyter Notebook:**
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=env
    ```

    **Command Prompt:**
    ```cmd
    pip install ipykernel
    python -m ipykernel install --user --name=env
    ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

    **Command Prompt:**
    ```cmd
    pip install -r requirements.txt
    ```

5. **Run Setup.py:**
    You can skip the 3rd and 4th step and directly run setup.py
    ```bash
    python setup.py
    ```
    **Command Prompt**
    ```Cmd
    python setup.py
    ```

## Usage
To use the Diamond Price Prediction project, follow these steps:

1. **Prepare the dataset:**
   Ensure you have the dataset in the `notebooks/data` directory and rename it to `gemstone.csv`.

2. **Run the training pipeline:**
    ```bash
    python src/pipelines/training_pipeline.py
    ```

3. **Run the prediction pipeline:**
    ```bash
    python src/pipelines/prediction_pipeline.py
    ```

4. **Start the application:**
    ```bash
    python app.py
    ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000/predict` to see the project running on your local host.

## Project Explanation
In this project, we aim to predict the price of diamonds based on the given dataset.

- **EDA.ipynb**: This file contains a detailed explanation of the dataset, along with analysis and visualizations using pandas, numpy, seaborn, and matplotlib.
- **model_training.ipynb**: In this notebook, the dataset is trained using multiple regression models from the scikit-learn library (`sklearn`). The best model is then selected and saved for prediction purposes.
- The project is built in a modular way, with separate scripts for data ingestion, transformation, and model training.
- Runtime logs can be found in the `logs` folder for further understanding of the processes.

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. **Fork the repository**
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

## Contact
If you have any questions or suggestions, feel free to reach out:

- **Email**: sagarpatelsp714@gmail.com.com
- **GitHub**: [SHP50c](https://github.com/SHP50c)

---

Thank you for visiting the Diamond Price Prediction project! We hope you find it useful and informative.
