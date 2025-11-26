# Data Collection in riotdb machine

![image](https://github.com/user-attachments/assets/03260b27-acc5-4ad8-adf4-c6d3aa89d9de)

# Cite this 

Plaintext

S. Barua and A. Baumgartner, "Weather-Aware LoRaWAN Sensor localization," Mobilkommunikation; 29. ITG-Fachtagung, Osnabrück, 2025, pp. 103-108.

Bibtext

````bash
@INPROCEEDINGS{11048384,
  author={Barua, Sudip and Baumgartner, Andreas},
  booktitle={Mobilkommunikation; 29. ITG-Fachtagung}, 
  title={Weather-Aware LoRaWAN Sensor localization}, 
  year={2025},
  volume={},
  number={},
  pages={103-108},
  keywords={},
  doi={}}
````

https://ieeexplore.ieee.org/abstract/document/11048384

<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/98c371e2-d6d1-4f0e-93cf-ff245cc1b4eb" />

## 📄 Weather-Aware LoRaWAN Sensor Localization

This document provides a summary and key details of the paper **"Weather-Aware LoRaWAN Sensor localization"** , which proposes an improved method for localizing LoRaWAN sensors by integrating climatic parameters into machine learning-based fingerprinting.

The work was presented at the ITG-Fachbericht 319: Mobilkommunikation conference.

---

### 🌟 Key Contributions

The primary contributions of this work are:

1.  A new approach of received-signal-strength (RSSI)-based fingerprinting comprising environmental information.
2.  A comparison of this new method against conventional mathematical position estimation and state-of-the-art machine learning techniques.

The proposed approach can improve state-of-the-art location methods by **23%** on the same dataset.

---

### 💡 Core Concept: Fingerprinting with Climate Data

Traditional GNSS-free localization methods for LoRaWAN, which operates in the sub-gigahertz spectrum, introduce non-negligible errors in localization estimation[cite: 9, 10, 11]. The proposed solution is to predict the coordinates of end devices from the pattern of signal strength variation and receiving gateway constellations at different environmental conditions.

The performance comparison in the paper shows the superior accuracy of the proposed weather-aware fingerprinting methods (FP-wx-RF and FP-wx-KNN):

| Method | Mean Error (meters) | Median Error (meters) |
| :--- | :--- | :--- |
| **FP-wx-RF (Proposed)** | **230**  | **112**  |
| FP-wx-KNN (Proposed) | 263  | 137  |
| FP-RF (Baseline) | 273  | 165  |
| FP-KNN (Baseline) | 301  | 185  |
| TDoA | 732  | 765  |
| PLM-OH (Okumura-Hata) | 1,700  | 1,080  |

---

### ⚙️ Feature Categories

The features used to train the machine learning models are categorized below[cite: 273, 239, 240, 242]:

| Category | Features Included |
| :--- | :--- |
| **Signal Metadata** | RSSI at receiving gateway, RX Time, SF, HDOP |
| **Position** | Latitude, Longitude |
| **General Weather** | Temperature, Relative Humidity, Dew Point, Apparent Temperature at 2m height |
| **Precipitation** | Precipitation, Rain, Snowfall, Snow Depth |
| **Pressure** | Pressure at sea level, Surface Pressure |
| **Clouds** | Cloud Cover (Total, Low, Mid, High) |
| **Evapotranspiration** | ETO FAO, Vapor Pressure Deficit |
| **Wind** | Wind Speed/Direction (10m, 100m), Wind Gusts 10m |
| **Radiation** | Shortwave, Direct, Diffuse, DNI, GTI, Terrestrial |

---

### 📈 Methodology & Workflow

The study's workflow, depicted in Figure 2, involves data acquisition, processing, estimation, and evaluation.

#### 1. Data Acquisition

* **LoRaWAN Data:** The study used an open-source dataset collected from Antwerp, Belgium in 2019 , containing 130,430 LoRaWAN messages from sensors utilizing a network of 72 gateways.
* **Weather Data:** Historical weather data was collected using Open-Meteo.
* **Correlation:** The LoRaWAN data was correlated with the historical weather data using GPS coordinates and timestamps.

#### 2. Processing

* The dataset was filtered to retain only samples received by **at least four gateways** , resulting in **55,303 data samples**.
* The **min-max scaling method** was used for normalization.

#### 3. Position Estimation (Training)

* The dataset was divided into a **70% training set** and a **30% testing set**.
* **5-fold cross-validation** was used to further generalize the models.
* **Algorithms:** The primary machine learning algorithms evaluated were **k-nearest neighbor (KNN) Regressor** and **Random Forest (RF) Regressor**.

#### 4. Model Evaluation

* Models were evaluated by calculating the **Haversine distance** between the predicted coordinates and the ground truth values.

---

### 👨‍🔬 Authors and Affiliation

* **Sudip Barua** 
    * Chair of Communication Networks, Technische Universität Chemnitz, Chemnitz, Germany 
* **Andreas Baumgartner** 
    * Chair of Communication Networks, Technische Universität Chemnitz, Chemnitz, Germany 

---

### 💰 Acknowledgement

This work was carried out as part of the **BSI-funded "Resilient Internet Of Things" (RIOT) project** (Grant ID: 01M023011C).

---
