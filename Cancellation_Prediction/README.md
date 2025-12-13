# Hotel Reservation Cancellation Prediction

This project predicts whether a hotel reservation will be canceled or not using a Machine Learning model. It includes a Streamlit web application for interactive predictions.

## Project Structure
- `Hotel_Reservation.ipynb`: Jupyter Notebook containing data analysis, preprocessing, and model training.
- `app.py`: Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `artifacts/`: Directory containing the trained model and other necessary files.

## Setup Instructions

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a Virtual Environment**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    - Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - Mac/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

To launch the Streamlit application, run the following command in your terminal (ensure your virtual environment is activated):

```bash
streamlit run app.py
```

The app should open automatically in your browser at `http://localhost:8501`.

## Usage
Enter the reservation details such as Lead Time, Average Price, Special Requests, etc., and click "Predict" to see the cancellation risk.
