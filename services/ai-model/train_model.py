import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import matplotlib.pyplot as plt

DATASET_NAME = "pet_vitals_v1.csv"
MODEL_PATH = "pet_health_model.pkl"
df = pd.read_csv(DATASET_NAME)

# Признаки и целевая переменная
X = df[['species', 'breed', 'heartRate', 'respiration', 'temperature']]
y = df['anomaly_class']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Предобработка
categorical_cols = ['species', 'breed']
numerical_cols = ['heartRate', 'respiration', 'temperature']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Модель
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Обучение
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:\n", cm)

# Для бинарной задачи (норма vs все аномалии)
y_test_bin = (y_test != 0).astype(int)   # 0 – норма, 1 – аномалия
y_pred_bin = (y_pred != 0).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()

type1_error = fp / (fp + tn)   # ошибка первого рода (False Positive Rate)
type2_error = fn / (fn + tp)   # ошибка второго рода (False Negative Rate)

print(f"Ошибка 1-го рода (ложная тревога, модель сказала «аномалия»): {type1_error:.4f}")
print(f"Ошибка 2-го рода (пропуск аномалии, модель сказала «норма»): {type2_error:.4f}")

print("Отчёт о классификации на тестовой выборке:")
print(classification_report(y_test, y_pred, target_names=['норма', 'аном.пульс', 'аном.дыхан.', 'аном.темп.']))

# Сохраняем модель временно локально
local_model_path = MODEL_PATH
joblib.dump(model, local_model_path)
print(f"Модель сохранена локально: {local_model_path}")

# Берём вероятности для класса "аномалия" (суммируем вероятности трёх аномальных классов)
y_pred_proba_anomaly = model.predict_proba(X_test)[:, 1:].sum(axis=1)  # если порядок классов [0,1,2,3]

fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba_anomaly)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate (ошибка 1-го рода)')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая: норма против аномалии')
plt.legend()
plt.show()