from src.data.make_dataset import load_raw_data
from src.features.build_features import preprocess_data
from src.models.train_model import train_lgb_model

if __name__ == "__main__":
    filepath = "data/raw/avocados.csv"
    model_path = "models/model_lgbm.pkl"

    df = load_raw_data(filepath)
    X, y = preprocess_data(df)
    rmse, r2 = train_lgb_model(X, y, model_path)

    print(f"✅ Model trained successfully\nRMSE: {rmse:.4f} | R²: {r2:.4f}")
