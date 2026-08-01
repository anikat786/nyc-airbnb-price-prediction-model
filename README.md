# 𝗡𝗬𝗖 𝗔𝗶𝗿𝗯𝗻𝗯 𝗣𝗿𝗶𝗰𝗲 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻 𝗠𝗼𝗱𝗲𝗹
Capstone project for the Machine Learning Foundations Course at eCornell and Break Through Tech AI. This project follows the full machine learning life cycle (problem definition, EDA, data preparation, and model training) to predict whether an Airbnb listing in New York City is "high-priced," comparing a traditional ML model against a neural network.

## 𝗣𝗿𝗼𝗷𝗲𝗰𝘁 𝗢𝘃𝗲𝗿𝘃𝗶𝗲𝘄
**Business problem:** Predict whether a given NYC Airbnb listing falls into the "high price" category (top 25% of prices) or "low price" category, based on listing, host, and review characteristics, not on price itself.
 
**Why it matters:** A model like this can help hosts understand which listing attributes are most associated with commanding a higher price, giving them concrete, data-driven guidance on how to price competitively and improve their listing's appeal.
 
**Data set:** `airbnbListingsData.csv`, NYC Airbnb listings with 51 original features covering listing details, host information, availability, and review scores.
 
**Label:** `price_category` (`high` / `low`), encoded to binary (`1` / `0`) for modeling. The 75th percentile of price is the cutoff.

## 𝗥𝗲𝗽𝗼𝘀𝗶𝘁𝗼𝗿𝘆 𝗦𝘁𝗿𝘂𝗰𝘁𝘂𝗿𝗲
This project is implemented as a single Jupyter notebook that walks through each stage of the ML life cycle:
 
| Part | Description |
|------|-------------|
| 1 | Load the data set and build the initial DataFrame |
| 2 | Define the ML problem (label, features, business context) |
| 3 | Exploratory data analysis: class balance, missing values, outliers, feature-label relationships, ethical considerations |
| 4 | Data preparation: cleaning, imputation, encoding, feature selection |
| 5 | Train, test, evaluate, and tune a traditional ML model (Logistic Regression) |
| 6 | Train, test, and evaluate a feedforward neural network (Keras) |
| 7 | Compare both models and reflect on results |

## 𝗠𝗲𝘁𝗵𝗼𝗱𝗼𝗹𝗼𝗴𝘆 𝗦𝘂𝗺𝗺𝗮𝗿𝘆

### 𝗗𝗮𝘁𝗮 𝗣𝗿𝗲𝗽𝗮𝗿𝗮𝘁𝗶𝗼𝗻
- Dropped free-text and leakage-prone columns (`name`, `description`, `neighborhood_overview`, `host_about`, `host_location`, and `price` itself)
- Removed `host_total_listings_count` as a confirmed duplicate of `host_listings_count`
- Imputed missing numeric values (`bedrooms`, `beds`, `host_response_rate`, `host_acceptance_rate`) using the median
- Converted boolean columns to numeric (0/1)
- One-hot encoded `room_type` and `neighbourhood_group_cleansed`
- Winsorized outliers in `host_listings_count` and `minimum_nights` using the IQR method
- Applied `StandardScaler` to all features
- Addressed class imbalance (about 74% low / 26% high) using `class_weight='balanced'` and evaluated with F1 score alongside accuracy

### 𝗠𝗼𝗱𝗲𝗹 𝟭: 𝗟𝗼𝗴𝗶𝘀𝘁𝗶𝗰 𝗥𝗲𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻
- Tuned via `GridSearchCV` over regularization strength (`C`) and `class_weight`
- Best hyperparameters: `C=0.1`, `class_weight='balanced'`
- Chosen for its interpretability: coefficients directly show how each feature relates to the odds of a high-priced listing

### 𝗠𝗼𝗱𝗲𝗹 𝟮: 𝗡𝗲𝘂𝗿𝗮𝗹 𝗡𝗲𝘁𝘄𝗼𝗿𝗸 (𝗞𝗲𝗿𝗮𝘀)
- Feedforward architecture: two hidden layers (64 and 32 units, ReLU activation), sigmoid output layer
- Optimizer: SGD, learning rate 0.01
- Loss: Binary cross-entropy
- Trained for 100 epochs with a 20% validation split

## 𝗥𝗲𝘀𝘂𝗹𝘁𝘀
| Metric | Logistic Regression | Neural Network |
|---|---|---|
| Accuracy | 0.783 | 0.845 |
| F1 Score | 0.665 | 0.694 |

The neural network outperformed logistic regression on both metrics, but the training curves showed signs of overfitting after roughly epoch 40 to 60 (validation loss began rising while training loss kept falling).

## 𝗥𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗮𝘁𝗶𝗼𝗻
Despite the neural network's stronger performance, **logistic regression is the recommended model for deployment** given the business context. The performance gap is modest, while logistic regression offers:
- Clear, interpretable coefficients hosts and stakeholders can act on
- Much faster training time
- Lower complexity to maintain and explain

## 𝗘𝘁𝗵𝗶𝗰𝗮𝗹 𝗖𝗼𝗻𝘀𝗶𝗱𝗲𝗿𝗮𝘁𝗶𝗼𝗻𝘀
The data set reflects existing NYC housing patterns, and `neighbourhood_group_cleansed` can act as a proxy for socioeconomic status. Incorrect predictions could disadvantage hosts in historically marginalized or lower-income neighborhoods (underpricing) or mislead hosts elsewhere into overpricing, reducing affordability for guests. These tradeoffs are discussed in more detail in the notebook.

## 𝗥𝗲𝗾𝘂𝗶𝗿𝗲𝗺𝗲𝗻𝘁𝘀
```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow (keras)
```
