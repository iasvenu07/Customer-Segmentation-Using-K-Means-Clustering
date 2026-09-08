# 📊 Customer Segmentation Using Machine Learning (K-Means Clustering)

An end-to-end **data science project and interactive web application** that segments retail/mall customers based on purchasing behavior and financial demographics using **unsupervised machine learning**.

The project combines a complete Python machine learning pipeline with an interactive single-page web dashboard for exploratory analysis, cluster tuning, customer persona profiling, and real-time customer segment prediction.

---

## ✨ Overview

Customer segmentation is a critical strategy in modern retail analytics. By categorizing shoppers based on attributes such as **Annual Income** and **Spending Score**, businesses can move from generic marketing campaigns to tailored, persona-driven commercial strategies that improve conversion opportunities and Customer Lifetime Value (CLTV).

This repository contains:

1. **Full Python Machine Learning Pipeline**  
   A Scikit-Learn implementation covering EDA, feature scaling, Elbow Method analysis, Silhouette Score evaluation, and K-Means++ model training.

2. **Interactive Single-Page Web Studio (`index.html`)**  
   A dashboard featuring dynamic data exploration, hyperparameter tuning visualization, interactive persona profiling, live customer segment prediction, and Python code documentation.

---

## 📁 Repository Structure

```text
customer-segmentation-kmeans/
│
├── index.html
│   # Single-page interactive web app & analytics dashboard
│
├── customer_segmentation_project.md
│   # Full step-by-step Python data science report
│
├── Mall_Customers.csv
│   # Source dataset (or auto-generated synthetic fallback)
│
└── README.md
   # Project documentation
```

---

## 🚀 Key Features

### 1. 🔍 Interactive Exploratory Data Analysis (EDA)

The dashboard provides dynamic exploratory analysis for:

- **Annual Income**
- **Spending Score**
- **Age**

It also provides real-time statistical metrics including:

- Mean
- Median
- Minimum
- Maximum
- Skewness

---

### 2. 🎛️ Live Cluster Hyperparameter Simulator

An interactive **K-Tuning** module allows users to simulate different cluster configurations.

Features include:

- Interactive slider for **k = 2 to 10**
- Real-time cluster selection
- Side-by-side **Elbow Curve (WCSS / Inertia)**
- **Silhouette Score** analysis
- Mathematical validation of the selected cluster count

The analysis identifies **k = 5** as the optimal number of clusters, with a Silhouette Score of approximately **0.554**.

---

### 3. 🎯 Interactive Segment Profiler & Playbook Explorer

Explore customer clusters through an interactive visual profiling interface.

Features include:

- 2D scatter plot visualization
- Cluster centroid mapping in scaled feature space
- Customer persona cards
- Demographic statistics
- Revenue/population share
- Recommended commercial strategies
- Persona-specific marketing playbooks

---

### 4. 🔮 Real-Time Customer Segment Predictor

The integrated inference form allows business stakeholders to enter customer attributes such as:

- **Annual Income**
- **Spending Score**

The application then classifies prospective customers into the appropriate cluster and provides an actionable marketing strategy.

This turns the clustering model from a static analysis exercise into something vaguely resembling an actual business tool, which is generally useful.

---

### 5. 🐍 In-App Python Code Explorer

The application includes a tabbed code viewer containing step-by-step Python walkthroughs for:

- Data preprocessing
- Feature scaling
- Exploratory analysis
- K-Means modeling
- Hyperparameter evaluation
- Visualization
- Customer segmentation

---

## 🎯 Identified Customer Personas (k = 5)

| Cluster ID | Persona Name | Avg Income (k$) | Avg Spending Score | Population Share | Recommended Commercial Strategy |
|---:|---|---:|---:|---:|---|
| **0** | **Core Customers** | $55.3k | 49.5 | ~40.5% | Focus on brand loyalty, subscription perks, and standard baseline offers. |
| **1** | **Careful Spenders** | $88.2k | 71.1 | ~17.5% | Target with high-value guarantees, premium quality items, and personalized campaigns. |
| **2** | **Budget Conscious** | $26.3k | 20.9 | ~11.5% | Engage through seasonal clearance sales, value packs, and price-match guarantees. |
| **3** | **Impatient Spenders** | $25.7k | 79.4 | ~11.0% | Deploy flash sales, impulse-buy bundles, and Buy-Now-Pay-Later (BNPL) options. |
| **4** | **Target / Premium** | $86.5k | 82.1 | ~19.5% | Offer VIP access, exclusive product launches, concierge rewards, and luxury events. |

> **Note:** Cluster labels are business personas assigned to the K-Means clusters to make the model outputs easier to interpret for marketing and commercial decision-making.

---

## 🧰 Quick Start Guide

### Running the Interactive Web App Locally

No web server setup is required for the basic dashboard.

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

#### 2. Open the web application

Open:

```text
index.html
```

directly in a modern browser.

Supported browsers include:

- Google Chrome
- Mozilla Firefox
- Microsoft Edge
- Safari

---

### Running the Python Data Science Pipeline

#### 1. Ensure Python 3.8+ is installed

Check your Python installation:

```bash
python --version
```

#### 2. Install the required dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

#### 3. Run the analysis

Execute the Python pipeline or follow the tutorial steps documented in:

```text
customer_segmentation_project.md
```

---

## 🔬 Machine Learning Methodology

The project follows the pipeline below:

```text
[Raw Dataset]
      ↓
[EDA & Univariate Plots]
      ↓
[StandardScaler Feature Scaling]
      ↓
[Hyperparameter Tuning: Elbow & Silhouette]
      ↓
[K-Means Model (k = 5)]
      ↓
[Inference / Customer Segment Prediction]
```

### 1. Preprocessing & Feature Scaling

Because K-Means relies on **Euclidean distance**, variables are normalized using `StandardScaler` to prevent features with larger numerical magnitudes from dominating the clustering process.

### 2. Clustering Algorithm

The project uses the K-Means algorithm with K-Means++ initialization:

```python
KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)
```

### 3. Model Validation

The clustering configuration is evaluated using:

- **Within-Cluster Sum of Squares (WCSS) / Inertia**
- **Silhouette Coefficient**

The metrics are evaluated across:

```text
k ∈ [2, 10]
```

The selected configuration is:

```text
Optimal k = 5
Silhouette Score ≈ 0.554
```

---

## 📊 Business Insights

The resulting customer personas provide a foundation for differentiated marketing strategies.

### Core Customers

A large baseline customer group with moderate income and spending behavior.

**Strategy:**

- Loyalty programs
- Subscription benefits
- Standard promotions
- Retention campaigns

### Careful Spenders

Customers with relatively high income who demonstrate controlled spending behavior.

**Strategy:**

- Premium product recommendations
- Personalized campaigns
- Quality-focused offers
- High-value guarantees

### Budget Conscious

Lower-income customers with lower spending scores.

**Strategy:**

- Discount campaigns
- Seasonal clearance
- Value bundles
- Price-match offers

### Impatient Spenders

Customers with lower income but high spending scores.

**Strategy:**

- Flash sales
- Impulse-buy bundles
- Limited-time offers
- BNPL options

### Target / Premium

High-income customers with high spending scores and strong commercial value.

**Strategy:**

- VIP programs
- Exclusive product launches
- Concierge services
- Luxury experiences
- Premium rewards

---

## 🛠️ Tech Stack

### Machine Learning & Analytics

- **Python**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

### Frontend UI & Visualization

- **HTML5**
- **Vanilla JavaScript**
- **Tailwind CSS (CDN)**
- **Chart.js (Canvas API)**

---

## 📌 Project Highlights

| Capability | Implementation |
|---|---|
| Machine Learning | K-Means Clustering |
| Initialization | K-Means++ |
| Feature Scaling | StandardScaler |
| Cluster Selection | Elbow Method + Silhouette Score |
| Optimal Clusters | k = 5 |
| Validation Score | ≈ 0.554 Silhouette Score |
| Dataset | Mall Customers |
| EDA | Interactive statistical and visual analysis |
| Prediction | Real-time customer segment inference |
| Dashboard | Single-page interactive web application |
| Visualization | Chart.js |
| Styling | Tailwind CSS |
| Documentation | Python walkthrough + README |

---

## 🔮 Future Enhancements

Potential extensions for the project include:

- Automatic cluster naming using LLM-assisted analysis
- Advanced customer lifetime value prediction
- RFM-based segmentation
- Automated marketing campaign recommendations
- Customer churn prediction
- Real-time database integration
- CSV/Excel dataset upload
- Model persistence using `joblib`
- REST API deployment
- Streamlit or FastAPI deployment
- Cloud deployment
- Advanced clustering comparison using DBSCAN and hierarchical clustering

---

## 📄 License

Distributed under the **MIT License**.

See the `LICENSE` file for more information.

---

## 👨‍💻 Author

**Your Name**

If you find this project useful, consider giving the repository a ⭐ on GitHub.
