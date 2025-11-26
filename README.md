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

<img width="959" height="775" alt="image" src="https://github.com/user-attachments/assets/98c371e2-d6d1-4f0e-93cf-ff245cc1b4eb" />

## 📄 Weather-Aware LoRaWAN Sensor Localization

[cite_start]This document provides a summary and key details of the paper **"Weather-Aware LoRaWAN Sensor localization"** [cite: 2][cite_start], which proposes an improved method for localizing LoRaWAN sensors by integrating climatic parameters into machine learning-based fingerprinting[cite: 12].

[cite_start]The work was presented at the ITG-Fachbericht 319: Mobilkommunikation conference[cite: 1, 41].

---

### 🌟 Key Contributions

[cite_start]The primary contributions of this work are[cite: 43]:

1.  [cite_start]A new approach of received-signal-strength (RSSI)-based fingerprinting comprising environmental information[cite: 44].
2.  [cite_start]A comparison of this new method against conventional mathematical position estimation and state-of-the-art machine learning techniques[cite: 45].

[cite_start]The proposed approach can improve state-of-the-art location methods by **23%** on the same dataset[cite: 14, 306].

---

### 💡 Core Concept: Fingerprinting with Climate Data

[cite_start]Traditional GNSS-free localization methods for LoRaWAN, which operates in the sub-gigahertz spectrum, introduce non-negligible errors in localization estimation[cite: 9, 10, 11]. [cite_start]The proposed solution is to predict the coordinates of end devices from the pattern of signal strength variation and receiving gateway constellations at different environmental conditions[cite: 228].

[cite_start]The performance comparison in the paper shows the superior accuracy of the proposed weather-aware fingerprinting methods (FP-wx-RF and FP-wx-KNN)[cite: 295, 303]:

| Method | Mean Error (meters) | Median Error (meters) |
| :--- | :--- | :--- |
| **FP-wx-RF (Proposed)** | [cite_start]**230** [cite: 247, 303] | [cite_start]**112** [cite: 255, 303] |
| FP-wx-KNN (Proposed) | [cite_start]263 [cite: 248, 303] | [cite_start]137 [cite: 255, 303] |
| FP-RF (Baseline) | [cite_start]273 [cite: 249, 303] | [cite_start]165 [cite: 256, 303] |
| FP-KNN (Baseline) | [cite_start]301 [cite: 250, 303] | [cite_start]185 [cite: 257, 303] |
| TDoA | [cite_start]732 [cite: 251, 304] | [cite_start]765 [cite: 258, 304] |
| PLM-OH (Okumura-Hata) | [cite_start]1,700 [cite: 253, 304] | [cite_start]1,080 [cite: 259, 304] |

---

### ⚙️ Feature Categories

[cite_start]The features used to train the machine learning models are categorized below[cite: 273, 239, 240, 242]:

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

[cite_start]The study's workflow, depicted in Figure 2[cite: 216, 229], involves data acquisition, processing, estimation, and evaluation.

#### 1. Data Acquisition

* [cite_start]**LoRaWAN Data:** The study used an open-source dataset collected from Antwerp, Belgium in 2019 [cite: 260][cite_start], containing 130,430 LoRaWAN messages from sensors utilizing a network of 72 gateways[cite: 261].
* [cite_start]**Weather Data:** Historical weather data was collected using Open-Meteo[cite: 262, 263].
* [cite_start]**Correlation:** The LoRaWAN data was correlated with the historical weather data using GPS coordinates and timestamps[cite: 264].

#### 2. Processing

* [cite_start]The dataset was filtered to retain only samples received by **at least four gateways** [cite: 267][cite_start], resulting in **55,303 data samples**[cite: 268].
* [cite_start]The **min-max scaling method** was used for normalization[cite: 270, 271].

#### 3. Position Estimation (Training)

* [cite_start]The dataset was divided into a **70% training set** and a **30% testing set**[cite: 275].
* [cite_start]**5-fold cross-validation** was used to further generalize the models[cite: 284].
* [cite_start]**Algorithms:** The primary machine learning algorithms evaluated were **k-nearest neighbor (KNN) Regressor** and **Random Forest (RF) Regressor**[cite: 12, 183].

#### 4. Model Evaluation

* [cite_start]Models were evaluated by calculating the **Haversine distance** between the predicted coordinates and the ground truth values[cite: 286, 287].

---

### 👨‍🔬 Authors and Affiliation

* [cite_start]**Sudip Barua** [cite: 3]
    * [cite_start]Chair of Communication Networks, Technische Universität Chemnitz, Chemnitz, Germany [cite: 4, 5]
* [cite_start]**Andreas Baumgartner** [cite: 6]
    * [cite_start]Chair of Communication Networks, Technische Universität Chemnitz, Chemnitz, Germany [cite: 7, 8]

---

### 💰 Acknowledgement

[cite_start]This work was carried out as part of the **BSI-funded "Resilient Internet Of Things" (RIOT) project** (Grant ID: 01M023011C)[cite: 312].

---

Would you like to know the specific formulas used for the **Time-Difference-of-Arrival (TDoA)** method described in the paper?
