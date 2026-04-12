import pandas as pd
import numpy as np

np.random.seed(42)

dog_breeds = {
    'Labrador': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
    'German Shepherd': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
    'Beagle': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
    'Poodle': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
    'Bulldog': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
    'Siberian Husky': {'hr_min': 60, 'hr_max': 100, 'rr_min': 10, 'rr_max': 30, 'temp_min': 37.5, 'temp_max': 39.0},
}

cat_breeds = {
    'Persian': {'hr_min': 120, 'hr_max': 220, 'rr_min': 16, 'rr_max': 40, 'temp_min': 38.0, 'temp_max': 39.2},
    'Siamese': {'hr_min': 120, 'hr_max': 220, 'rr_min': 16, 'rr_max': 40, 'temp_min': 38.0, 'temp_max': 39.2},
    'Maine Coon': {'hr_min': 120, 'hr_max': 220, 'rr_min': 16, 'rr_max': 40, 'temp_min': 38.0, 'temp_max': 39.2},
    'Bengal': {'hr_min': 120, 'hr_max': 220, 'rr_min': 16, 'rr_max': 40, 'temp_min': 38.0, 'temp_max': 39.2},
    'Sphynx': {'hr_min': 120, 'hr_max': 220, 'rr_min': 16, 'rr_max': 40, 'temp_min': 38.0, 'temp_max': 39.2},
}

def generate_record(species, breed, params):
    heartRate = np.random.uniform(params['hr_min'], params['hr_max'])
    respiration = np.random.uniform(params['rr_min'], params['rr_max'])
    temperature = np.random.uniform(params['temp_min'], params['temp_max'])
    
    anomaly_class = 0
    
    # Вероятность аномалии 5%
    if np.random.rand() < 0.05:
        anomaly_type = np.random.choice([1, 2, 3])
        if anomaly_type == 1:  # пульс
            heartRate = np.random.uniform(params['hr_max'] * 1.2, params['hr_max'] * 2.0)
            anomaly_class = 1
        elif anomaly_type == 2:  # дыхание
            respiration = np.random.uniform(params['rr_max'] * 1.5, params['rr_max'] * 3.0)
            anomaly_class = 2
        else:  # температура
            temperature = np.random.uniform(params['temp_max'] + 0.5, params['temp_max'] + 2.0)
            anomaly_class = 3
    
    return {
        'species': species,
        'breed': breed,
        'heartRate': round(heartRate, 1),
        'respiration': round(respiration, 1),
        'temperature': round(temperature, 1),
        'anomaly_class': anomaly_class  # 0-норма, 1-пульс, 2-дыхание, 3-температура
    }

records = []
for _ in range(2500):
    breed = np.random.choice(list(dog_breeds.keys()))
    params = dog_breeds[breed]
    records.append(generate_record('dog', breed, params))

for _ in range(2500):
    breed = np.random.choice(list(cat_breeds.keys()))
    params = cat_breeds[breed]
    records.append(generate_record('cat', breed, params))

np.random.shuffle(records)
df = pd.DataFrame(records)

print(df['anomaly_class'].value_counts())
df.to_csv('pet_vitals_v1.csv', index=False)