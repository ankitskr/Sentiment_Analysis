# Twitter Sentiment Analysis (Under Development)

## Project Overview
This project demonstrates the use of machine learning classification models to perform sentiment analysis on Twitter data. The aim is to classify tweets into positive or negative sentiments. We utilize various models from the Scikit-Learn library to accomplish this task.

## Project Structure
- **EDA and Data Preprocessing**: Initial exploration and preprocessing of data including cleaning, tokenization, and normalization.
- **Feature Extraction**: Conversion of text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
- **Model Selection and Training**: Various classification models are trained and their performance compared.
- **Hyperparameter Tuning**: Utilization of grid search for optimal model parameters based on weighted F1 score.
- **Threshold Adjustment**: Determination of the optimal threshold for maximizing the weighted F1 score.
- **Model Evaluation**: Final evaluation of models using a hold-out test set with metrics such as accuracy, precision, recall, and F1-score.


## Technologies Used
This project leverages several technologies and libraries within the Python ecosystem:

- **Python**: The primary programming language used.
- **NumPy and Pandas**: For numerical operations and data manipulation.
- **Matplotlib**: For data visualization.
- **Scikit-Learn**: Provides tools for advance data analysis, including various machine learning models.
- **NLTK**: For processing textual data.
- **Jupyter Notebook**: Used for creating and sharing the project notebook with live code.


## Working demonstration
A working demonstration is provided in the jupyter notebook.
- Model Evaluation: [notebooks/model_evaluation.ipynb](https://github.com/ankitskr/Twitter-Sentiment-Analysis/blob/master/notebooks/model_evaluation.ipynb)


## Results
The models achieve an accuracy of approximately 75%. Adjustments in model hyperparameters and threshold values were explored, showing minimal impact on overall model performance.

## Conclusion
The project encapsulates a full pipeline of processing and analyzing Twitter data for sentiment analysis using machine learning, providing a foundation for further exploration and refinement of classification models.