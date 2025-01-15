
# Fake News Detection

This project is a machine learning application to detect fake news articles using logistic regression and TF-IDF vectorization. The dataset used is **WELFake_Dataset.csv**, and the model classifies news articles as either "REAL" or "FAKE."

---

## Features
- Preprocessing text data with TF-IDF Vectorization
- Logistic Regression for classification
- Performance metrics: Accuracy, Classification Report, and Confusion Matrix
- Predict new text samples for fake/real classification

---

## Technologies Used
- Python
- Pandas
- Scikit-learn

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NivethaSre/fake-news-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fake-news-detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Prepare Dataset:** Ensure the `WELFake_Dataset.csv` file is placed in the correct directory (adjust the `dataset_path` in the script accordingly).

2. **Run the Script:**
    ```bash
    python fake_news_detection.py
    ```

3. **Test with a New Sample:**
    Modify the `sample_news` variable in the script to test custom news samples.

---

## Dataset

The project uses the `WELFake_Dataset.csv` file. Ensure the dataset includes:
- `text`: The content of the news article.
- `label`: The label indicating whether the news is "REAL" (1) or "FAKE" (0).

---

## Example

### Sample Output:
```plaintext
Accuracy: 92.35%

Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.90      0.92       500
           1       0.91      0.95      0.93       500

Confusion Matrix:
[[450  50]
 [ 25 475]]

The news article is predicted as: REAL
```







