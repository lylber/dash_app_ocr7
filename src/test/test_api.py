import pytest
from dash.testing.browser import Browser
from dash.testing.application_runners import import_app

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dash.testing.application_runners import import_app
import subprocess
import time
import sys
sys.path.append(r'src\\')
from ressources.dico_features import colonnes_catégoriques
#### Pour lancer et faciliter le test mettre des valeurs par défauts pour les variables catégories dans le script API sous dcc.Dropdown (code en commentaire) ###
d=pd.read_csv(r'C:\Users\Hilbert\Documents\OpenClassRoom\app_dash_ocr7\src\datasets\brut_train.csv')
d2=pd.read_csv(r'C:\Users\Hilbert\Documents\OpenClassRoom\app_dash_ocr7\src\datasets\train.csv').drop(['TARGET'],axis=1)
d2[colonnes_catégoriques]=d[colonnes_catégoriques]
exemple_client=dict(zip(d2.columns,d2.loc[0]))

options = webdriver.chrome.options.Options()
options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def start_dash_server():
    subprocess.Popen(["python", "src\API.py"])
    time.sleep(5)

start_dash_server()

@pytest.fixture(scope='module')
def app():
    app = import_app("API")
    app.config.suppress_callback_exceptions = True
    return app

@pytest.fixture(scope='module')
def driver():
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

def test_layout(app, driver):
    with app.server.app_context():
        driver.get("http://localhost:8050/")
        header_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.TAG_NAME, "h1"))
        )
        assert header_element.text.lower() == "prédiction modèle"

def test_prediction(app, driver):
    with app.server.app_context():
        driver.get("http://localhost:8050/")

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "predict-button"))
        )

        feature_names = [col.get_attribute("id").split('-')[1] for col in driver.find_elements(By.CSS_SELECTOR, "[id^='feature-']")]


        for  col in feature_names:
            input_element = driver.find_element(By.ID, f"feature-{col}")
            input_element.send_keys(str(exemple_client[col]))

        predict_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "predict-button"))
        )
        predict_button.click()

        # Attendre que le résultat de la prédiction soit affiché
        prediction_output = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, "prediction-output"))
        )
        proba=float(prediction_output.text.split(':')[-1])
    
        assert 0<= proba <= 1
        assert "La prédiction du modèle est" in prediction_output.text




 
