## What this project does

- Takes banana images across multiple days (time progression)  
- Predicts a **continuous ripeness score** between 0 and 1  
- Incorporates a **physics-based constraint** to guide learning  

---

## Run Locally

This project now includes a Flask backend and a served frontend UI.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python app.py
```

3. Open the frontend in your browser:

```text
http://127.0.0.1:5000/
```

The prediction API is available at:

```text
POST /predict
```

Send:

- `image` as form-data file
- `day` as an integer from `1` to `6`

Instead of treating ripeness as:
Unripe / Ripe / Rotten

we model it as:

$$
R(t) \in [0,1]
$$

---

## Core Idea

We learn a function:

$$
R = f(\text{image}, t)
$$

Where:

- **image** → visual features (color, texture, spots)  
- **t** → time (derived from day-wise dataset)  

We also enforce a simple biological assumption:

$$
\frac{dR}{dt} = k(1 - R)
$$

This helps the model learn **smooth and consistent ripeness progression** over time.

---

## Dataset

The dataset is organized as:


Day 1/
Day 2/
Day 3/
Day 4/
Day 5/
Day 6/


Each folder contains banana images captured on that day.

Time is normalized as:

$$
t = \frac{\text{day}}{6}
$$

For simplicity, the target ripeness is currently defined as:

$$
R = t
$$

This keeps the setup straightforward and allows focus on testing the PINN framework.

---

## 🏗 Model Setup

### CNN (Baseline)
- Takes only image as input  
- Predicts ripeness directly  

### PINN Model
- Takes **image + time** as input  
- Predicts ripeness  
- Uses autograd to compute $$\frac{dR}{dt}$$  
- Applies physics-based loss  

---

## ⚙️ Loss Function

$$
\text{Total Loss} = \text{Data Loss} + \lambda \times \text{Physics Loss}
$$

- **Data Loss** → Mean Squared Error between predicted and actual ripeness  
- **Physics Loss** → Enforces the differential equation  

---

## 🔬 What is implemented

- CNN baseline model  
- CNN + PINN model  
- Physics-informed loss using autograd  
- Learnable parameter $$k$$  
- Comparison between CNN and PINN  
- Basic visualization of predictions  

---

## Current Limitations
 
- Uses a simple first-order kinetic model  
- Limited dataset (6 days only)  

This is a **proof-of-concept**, not a full biological model.

---

## Why this is interesting

- Demonstrates integration of **deep learning and differential equations**  
- Introduces **domain knowledge into training**  
- Moves beyond standard classification-based approaches  

---

## Possible Improvements

- Use real ripeness measurements instead of $$R = t$$  
- Incorporate temperature or storage conditions  
- Expand dataset for better generalization
