import json
import pickle
from typing import List, Union

import pandas as pd


class RiskClassifier:
    """Class used to classify the risk of a client"""

    def __init__(self, models_dir: str = 'models/xgb_model.pkl', encoder_dir: str = 'models/target_encoder.json'):
        """Initialize RiskClassifier

        Args:
            models_dir (str): Path to model.
        """
        self.r_columns = [
            'acc_status',
            'duration_months',
            'credit_history',
            'credit_amount',
            'savings',
            'present_employment',
            'installment_rate',
            'personal_sex_status',
            'guarantors',
            'properties',
            'other_installment_plans',
            'foreign',
        ]
        self.models_dir = models_dir
        self.encoder_dir = encoder_dir
        self.model, self.encoder_json = self.load_model_and_preparation_objects()

    def __call__(self, x_features: List[Union[float, int, str]]) -> int:
        """Predict if a candidate has risk or not

        Args:
            x_features (List[Union[float, int, str]]): Input features

        Returns:
            int: 0 good candidate, 1 bad candidate
        """
        df_X = self.preprocess(x_features)
        return self.predict(df_X)

    def load_model_and_preparation_objects(self):
        """Load model and encoder

        Returns:
            Tuple[xgboost.sklearn.XGBClassifier, json]: Model and feature encoder
        """
        model = pickle.load(open(self.models_dir, 'rb'))
        with open(self.encoder_dir, 'r') as json_file:
            json_encoder = json.load(json_file)
        return model, json_encoder

    def preprocess(self, x_features: List[Union[float, int, str]]) -> pd.DataFrame:
        """Preprocess features

        Args:
            x_features (List[Union[float, int, str]]): Features

        Returns:
            pd.DataFrame: DataFrame with preprocessed features
        """

        df_X = pd.DataFrame([x_features], columns=self.r_columns)
        for k, v in self.encoder_json.items():
            if k in self.r_columns:
                df_X[k] = df_X[k].map(v)
        return df_X

    def predict(self, df_X: pd.DataFrame) -> int:
        """Predict a sample

        Args:
            df_X (pd.DataFrame): Preprocessed features

        Returns:
            int: 0 good candidate, 1 bad candidate
        """
        return self.model.predict(df_X)[0]


if __name__ == '__main__':
    risk_o = RiskClassifier()
    prediction = risk_o(['A11', 6, 'A34', 1169, 'A65', 'A75', 4, 'A93', 'A101', 'A121', 'A143', 'A201'])
    mapping = {0: 'Good Candidate', 1: 'Bad Candidate'}
    print(mapping[prediction])
