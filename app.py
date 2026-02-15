from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import date, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# ===== ファイルパス =====
TRAIN_CSV_PATH = os.path.join("data", "whistler_train.csv")    # 2020〜2026など（学習用）
SEASON_CSV_PATH = os.path.join("data", "whistler_season.csv")  # 2024/12〜2025/5など（実測表示用）

DATE_COL = "Date/Time"
TARGET_COL = "Total Snow (cm)"

# 未来予測（初心者向け：日付情報だけ）
FEATURE_COLS = ["day_number", "month","day"]


# -----------------------------
# 実測（合計・最大日・月ごと）を作る
# -----------------------------
def calc_season_stats():
    df = pd.read_csv(SEASON_CSV_PATH)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 2024/12〜2025/05 に絞る（必要ならここを変える）
    df = df[(df[DATE_COL] >= "2024-12-01") & (df[DATE_COL] <= "2025-05-31")]
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if len(df) == 0:
        return 0.0, "-", 0.0, []

    total = float(df[TARGET_COL].sum())

    max_row = df.loc[df[TARGET_COL].idxmax()]
    max_date = max_row[DATE_COL].strftime("%Y-%m-%d")
    max_snow = float(max_row[TARGET_COL])

    monthly = df.groupby(df[DATE_COL].dt.month)[TARGET_COL].sum().reset_index()
    monthly.columns = ["month", "snow"]
    monthly["snow"] = monthly["snow"].round(1)

    return round(total, 1), max_date, round(max_snow, 1), monthly.to_dict(orient="records")


# -----------------------------
# 学習してモデルを作る
# -----------------------------
def build_model():
    df = pd.read_csv(TRAIN_CSV_PATH)

    # 必須列チェック
    for col in [DATE_COL, TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"CSVに '{col}' 列が見つからないよ。列名を確認してね！")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    START_DATE = df[DATE_COL].min()
    df["day_number"] = (df[DATE_COL] - START_DATE).dt.days
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    
    model = LinearRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    last_train_date = df[DATE_COL].max().date()

    return model, float(mae), START_DATE, last_train_date


# 起動時に1回だけ学習
model, mae, START_DATE, LAST_TRAIN_DATE = build_model()


def predict_snow(target_date: date) -> float:
    """target_date の降雪量を予測（cm）"""
    day_number = (pd.Timestamp(target_date) - START_DATE).days
    month = target_date.month
    day = target_date.day

    X_pred = pd.DataFrame([[day_number, month,day]], columns=FEATURE_COLS)
    y_pred = float(model.predict(X_pred)[0])
    y_pred = max(0.0, y_pred)
    return round(y_pred, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    # 予測
    pred_date = None
    pred_snow = None
    error = None

    # デフォルトは「明日」
    default_date = date.today() + timedelta(days=1)
    date_value = default_date.strftime("%Y-%m-%d")

    if request.method == "POST":
        input_date = (request.form.get("date") or "").strip()
        if input_date == "":
            input_date = date_value

        try:
            target = pd.to_datetime(input_date, errors="raise").date()
            pred_snow = predict_snow(target)
            pred_date = target.strftime("%Y-%m-%d")
            date_value = pred_date
        except Exception:
            error = "日付は YYYY-MM-DD で入力してね（例：2026-02-13）"

    return render_template(
        "index.html",
        # 予測
        pred_date=pred_date,
        pred_snow=pred_snow,
        error=error,
        date_value=date_value,
        mae=round(mae, 3),
        last_train_date=str(LAST_TRAIN_DATE),

    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)