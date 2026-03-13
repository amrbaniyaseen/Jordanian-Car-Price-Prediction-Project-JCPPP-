from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = joblib.load('car_price_model.pkl')
except Exception:
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)

# ── Hard-coded training stats (from notebook) ─────────────────────────────────
# Brand median log-prices (approximate – replace with your brandMedians.csv values)
# These are loaded from CSV files that were saved during training.
# For portability, we embed the key lookup tables directly here.

BRAND_MEDIANS = {'Audi': 9.44931201340785, 'Avatr': 10.352989997453784, 'BAIC': 9.410984574215794,
                 'BMW': 9.741027444837728, 'BYD': 9.581972891547895, 'Bentley': 10.857093336791094,
                 'Bestune': 9.433563920090563, 'Buick': 8.77971129020447, 'Cadillac': 8.935948546754442,
                 'Changan': 9.358846580275406, 'Chery': 8.123760920629286, 'Chevrolet': 8.699681400989514,
                 'Chrysler': 9.174696017737965, 'Citroen': 7.9722163103502375, 'Daewoo': 7.170888478512505,
                 'Daihatsu': 8.03947991910045, 'Dodge': 9.615838812306619, 'Dongfeng': 9.392661928770137,
                 'Fiat': 8.594339400592892, 'Ford': 9.259225769705994, 'Forthing': 9.762819346464951,
                 'GAC': 9.680406499268875, 'GMC': 8.630700432209832, 'Geely': 9.465060106725137,
                 'Genesis': 9.350189267092581, 'Haval': 10.106993758577321,
                 'Honda': 8.665785595466064, 'Hongqi': 10.525411548278893,
                 'Hyundai': 8.748463629942055, 'Infiniti': 9.44943600950432,
                 'Isuzu': 8.968375911155523, 'JAC': 9.472781556562168,
                 'JMC': 9.425532393502772, 'JMEV': 9.295308378625812, 'Jaguar': 9.227640114174832,
                 'Jeep': 9.510519035757246, 'Jetour': 10.315630315397533, 'Kia': 8.81001204797317,
                 'Lada': 8.740496729931813, 'Land Rover': 9.504308701580165, 'Leapmotor': 9.176572339332463,
                 'Lexus': 9.38437770918244, 'Lincoln': 9.437547687480128, 'MG': 9.472781556562168,
                 'MINI': 9.200391041122515, 'Mahindra': 8.807056448127609,'Maserati': 10.455963627557306,
                 'Mazda': 8.821135430108754, 'Mercedes Benz': 10.021315031649332,
                 'Mercury': 8.955060883983016, 'Mitsubishi': 8.699681400989514, 'Neta': 9.564582358673212,
                 'Nissan': 8.682877107057168, 'Opel': 7.601402334583733, 'Peugeot': 8.03947991910045,
                 'Polestar': 9.841665338892785, 'Porsche': 9.741027444837728, 'Proton': 8.962007209588313,
                 'Renault': 8.05434078864678, 'Rising': 10.021315031649332, 'SAIPA': 9.305741456739435,
                 'Saab': 8.160803920954665, 'Samsung': 9.024131268455035, 'Saturn': 8.38958706681109,
                 'Seat': 7.5757688385360815, 'Seres': 9.726213236963687, 'Skoda': 7.3783837129967145,
                 'Skywell': 9.770013301136158, 'Smart': 8.366602832783736, 'SsangYong': 8.517393171418904,
                 'Subaru': 8.647781139966312, 'Suzuki': 8.366602832783736, 'Tesla': 9.867912187022208,
                 'Toyota': 9.305741456739435, 'Volkswagen': 9.388570235866881, 'Volvo': 8.006700845440367,
                 'Wuling': 8.656183286932485}
GLOBAL_BRAND_MEDIAN = 9.2

MODEL_MEDIANS = {'2': 8.556606193773073, '206': 7.741099090035366, '207': 8.256003839247953, '3': 8.86953877719374,
                 '3 Series': 9.259225769705994, '307': 7.928725768092342, '4 Series': 10.55843949396453,
                 '407': 8.071218539969863, '5 Series': 9.689737246238513, '500': 8.603512286734176,
                 '6': 9.160370537791696, '7 Series': 10.257523305886963, 'A6': 9.04793908261736,
                 'Acadia': 8.739696345596991, 'Accent': 7.824445930877619, 'Accord': 9.539301889815091,
                 'Altima': 8.648396877031582, 'Alto': 8.24301946898925, 'Astra': 8.294299608857235,
                 'Atos': 7.901377353792616, 'Avalon': 9.472781556562168, 'Avante': 8.188966863648876,
                 'Aveo': 8.09434932607855, 'Blazer': 8.31868874001331, 'Bolt': 9.648659816955295,
                 'Bora': 9.38856148390694, 'C-Class': 9.38437770918244, 'C-HR': 9.752655199826654,
                 'C-MAX': 9.116139576577355, 'CR-V': 8.612685172875459, 'CT': 9.259225769705994,
                 'Caddy': 8.853808274977197, 'Camry': 9.350189267092581, 'Captiva': 9.305741456739435,
                 'Cayenne': 9.15915233520675, 'Cerato': 8.881975184248867, 'Charger': 9.105090961257085,
                 'Civic': 8.412054873292933, 'Clio': 7.741099090035366, 'Corolla': 8.94910546953925,
                 'Cruze': 8.536999682595988, 'D-Max': 9.071193240566025, 'Destroyer 05': 9.539716058977092,
                 'Dolphin': 9.472781556562168, 'E-Class': 10.014603821770125, 'E-HS9': 10.545367754151743,
                 'E-Star': 9.04793908261736, 'E2': 9.53249633228804, 'ES': 9.637294037507324,
                 'EV6': 10.094149233605084, 'EX1': 9.08542108374738, 'Eado': 9.375939552625749,
                 'Eado EV': 9.314790473332646, 'Elantra': 8.81001204797317, 'Envoy': 8.412054873292933,
                 'Escape': 9.172138729020935, 'Explorer': 8.846640813100485, 'F-150': 9.741027444837728,
                 'Focus': 8.674331351261616, 'Forte': 8.674331351261616, 'Fusion': 9.259225769705994,
                 'GT': 9.94755228369823, 'Galant': 8.006700845440367, 'Gladiator': 10.370387723054154,
                 'Golf': 8.987321812850125, 'Golf MK': 7.696667081526462, 'Grand Cherokee': 9.390659926099465,
                 'Grand Vitara': 8.517393171418904, 'H 100': 7.937731775260109, 'H1': 8.853808274977197,
                 'HS': 9.074025415277093, 'Han': 10.026839705553172, 'Highlander': 10.305647203635566,
                 'Hilux': 9.69590972432601, 'ID 3': 9.6678285081515, 'ID 4': 9.928228944424575,
                 'ID 6': 10.05195066017375, 'Insight': 9.521507684022644, 'Ioniq': 9.510519035757246,
                 'K3': 9.21039036947635, 'K5': 9.358846580275406, 'Kadett': 7.00397413672268,
                 'Kona': 9.648659816955295, 'L200': 9.392745258631441, 'Lancer': 8.51226902338883,
                 'Land Cruiser': 10.757924157061597, 'Lanos': 7.313886831633462, 'Lavida': 9.392745258631441,
                 'LeMans': 7.00397413672268, 'Leaf': 8.665785595466064, 'Leopard 5': 10.645448706505872,
                 'Lotze': 8.306642867124648, 'MKZ': 9.44931201340785, 'Menlo': 9.664533158112086,
                 'Model 3': 9.813335582260542, 'Model S': 10.021315031649332, 'Model X': 10.275085591132715,
                 'Model Y': 10.021315031649332, 'Morning': 8.839421607620602, 'Mustang': 9.758519582283522,
                 'NX': 10.114074262056176, 'Nammi Box': 9.409273194575334, 'Navara': 9.14851476772461,
                 'Niro': 9.6678285081515, 'Niro EV': 9.738077757797978, 'Nubira': 7.279414188985234,
                 'Omega': 7.824445930877619, 'Optima': 9.105090961257085, 'Optra': 7.8636512654486515,
                 'Other': 8.987321812850125, 'Outlander': 8.922791623969637, 'Pajero': 9.19023970026918,
                 'Partner': 8.412054873292933, 'Passat': 8.77971129020447, 'Pathfinder': 9.392745258631441,
                 'Picanto': 8.69127925402334, 'Porter': 9.04793908261736, 'Prado': 9.210440366976517,
                 'Pregio': 7.696667081526462, 'Prius': 9.04793908261736, 'Qin': 9.642187721358466,
                 'RAV 4': 9.99884318585288, 'RX': 8.81001204797317, 'Ram': 9.648659816955295,
                 'Rio': 8.556606193773073, 'Santa Fe': 8.575650760987806, 'Seagull': 9.375939552625749,
                 'Seal': 9.581972891547895, 'Sephia': 7.3783837129967145, 'Shuma': 7.438971592395862,
                 'Sierra': 10.221977646629885, 'Silverado': 9.108781423266322, 'Sonata': 9.04793908261736,
                 'Song L': 10.106469212026385, 'Song Plus': 10.075379913839543, 'Song Pro': 9.943957485089841,
                 'Sorento': 9.259225769705994, 'Soul': 8.987321812850125, 'Spark': 8.471344954858932,
                 'Spectra': 7.901377353792616, 'Sportage': 9.082620630373812, 'Sunny': 8.366602832783736,
                 'T2': 10.315630315397533, 'Tahoe': 9.305741456739435, 'Tang': 10.404293143019792,
                 'Tucson': 8.853808274977197, 'U': 9.54649107666534, 'V': 9.164401140034737,
                 'Vectra': 7.550135342488429, 'Veloster': 8.935948546754442, 'Verna': 8.055349229535157,
                 'X-Trail': 9.15915233520675, 'X5 Series': 10.06457187367786, 'Yaris': 8.999666594943267,
                 'Yuan': 9.770013301136158, 'bZ': 9.90353755128617, 'e:N': 9.729193687429403,
                 'i10': 8.35490952835879, 'iX Series': 10.389026137075234}
GLOBAL_MODEL_MEDIAN = 9.2

RARE_MODELS = set()  # populated from training; we keep empty for inference

SCALER_MEAN  = np.array([10.0, 9.2, 9.2])   # Car Age, Brand Value, Model Value
SCALER_SCALE = np.array([8.0,  0.7, 0.7])


def create_scaler():
    """Recreate the StandardScaler using the saved mean/scale."""
    scaler = StandardScaler()
    scaler.mean_  = SCALER_MEAN
    scaler.scale_ = SCALER_SCALE
    scaler.var_   = SCALER_SCALE ** 2
    scaler.n_features_in_ = 3
    return scaler


SCALER = create_scaler()

MODEL_COLUMNS = [
    'Battery Capacity', 'Battery Range', 'Body Condition', 'Car License',
    'Condition', 'Insurance', 'Mileage', 'Paint', 'Car Age',
    'Brand Value', 'Model Value',
    'Fuel Type_Diesel', 'Fuel Type_Electric', 'Fuel Type_Gasoline',
    'Fuel Type_Hybrid', 'Fuel Type_Mild Hybrid', 'Fuel Type_Plug-in Hybrid',
    'Transmission_Automatic', 'Transmission_Manual',
    'Engine Size (cc)_0', 'Engine Size (cc)_0 - 499 cc',
    'Engine Size (cc)_1,000 - 1,999 cc', 'Engine Size (cc)_2,000 - 2,999 cc',
    'Engine Size (cc)_3,000 - 3,999 cc', 'Engine Size (cc)_4,000 - 4,999 cc',
    'Engine Size (cc)_5,000 - 5,999 cc', 'Engine Size (cc)_500 - 999 cc',
    'Engine Size (cc)_More than 6,000 cc',
    'Body Type_Bus', 'Body Type_Convertible', 'Body Type_Coupe',
    'Body Type_HatchBack', 'Body Type_PickUp', 'Body Type_SUV',
    'Body Type_Sedan', 'Body Type_Truck',
    'Car Customs_With Customs', 'Car Customs_Without Customs',
    'Market Specifications_American Specs', 'Market Specifications_Chinese Specs',
    'Market Specifications_European Specs', 'Market Specifications_GCC Specs',
    'Market Specifications_Japanese Specs', 'Market Specifications_Korean Specs',
    'Market Specifications_Other Specs',
]


def preprocess_car_input(user_input: dict) -> pd.DataFrame:
    """Replicate the exact ETL pipeline from the training notebook."""

    battery_capacity_map = {
        '0': 0, 'N/A': 0,
        'Less than 50 kWh': 1, '50 - 69 kWh': 2,
        '70 - 89 kWh': 3, '90 - 99 kWh': 4, 'More than 100 kWh': 5
    }
    battery_range_map = {
        '0': 0, 'N/A': 0,
        'Less than 100 km': 1, '100 - 199 km': 2, '200 - 299 km': 3,
        '300 - 399 km': 4, '400 - 499 km': 5, 'More than 500 km': 6,
    }
    mileage_map = {
        '0 km': 22, '1 - 999 km': 21, '1,000 - 9,999 km': 20,
        '10,000 - 19,999 km': 19, '20,000 - 29,999 km': 18,
        '30,000 - 39,999 km': 17, '40,000 - 49,999 km': 16,
        '50,000 - 59,999 km': 15, '60,000 - 69,999 km': 14,
        '70,000 - 79,999 km': 13, '80,000 - 89,999 km': 12,
        '90,000 - 99,999 km': 11, '100,000 - 109,999 km': 10,
        '110,000 - 119,999 km': 9, '120,000 - 129,999 km': 8,
        '130,000 - 139,999 km': 7, '140,000 - 149,999 km': 6,
        '150,000 - 159,999 km': 5, '160,000 - 169,999 km': 4,
        '170,000 - 179,999 km': 3, '180,000 - 189,999 km': 2,
        '190,000 - 199,999 km': 1, '+200,000 km': 0,
    }
    body_condition_map = {
        'Poor': 0, 'Fair': 1, 'Other': 2, 'Good': 3, 'Excellent': 4
    }
    paint_map = {
        'Total repaint': 0, 'Other': 1, 'Partially repainted': 2, 'Original Paint': 3
    }
    car_license_map = {'Not Licensed': 0, 'Licensed': 1}
    condition_map   = {'Used': 0, 'New': 1}
    insurance_map   = {
        'Not Insured': 0, 'Compulsory Insurance': 1, 'Comprehensive Insurance': 2
    }

    # Build single-row DataFrame
    x = pd.DataFrame([user_input])

    # Ordinal / label encoding
    ordinal_maps = {
        'Battery Capacity': battery_capacity_map,
        'Battery Range':    battery_range_map,
        'Body Condition':   body_condition_map,
        'Car License':      car_license_map,
        'Condition':        condition_map,
        'Insurance':        insurance_map,
        'Mileage':          mileage_map,
        'Paint':            paint_map,
    }
    for col, mapping in ordinal_maps.items():
        x[col] = x[col].map(mapping).fillna(0)

    # Car Age
    current_year = datetime.now().year
    x['Car Age'] = current_year - x['Model Year'].astype(int)

    # Target encoding
    x['Brand Value'] = x['Car Make'].map(BRAND_MEDIANS).fillna(GLOBAL_BRAND_MEDIAN)
    x['Model Value'] = x['Model'].map(MODEL_MEDIANS).fillna(GLOBAL_MODEL_MEDIAN)

    x = x.drop(columns=['Car Make', 'Model', 'Model Year'], errors='ignore')

    # One-hot encoding
    one_hot_cols = [
        'Body Type', 'Car Customs', 'Engine Size (cc)',
        'Fuel Type', 'Transmission', 'Market Specifications'
    ]
    x = pd.get_dummies(x, columns=one_hot_cols, dtype=int)

    # Align columns to training schema
    x = x.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # Scale numerical columns
    scale_cols = ['Car Age', 'Brand Value', 'Model Value']
    x[scale_cols] = SCALER.transform(x[scale_cols])

    return x


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Validate required fields
        required = [
            'Battery Capacity', 'Battery Range', 'Body Condition', 'Body Type',
            'Car Customs', 'Car License', 'Car Make', 'Condition',
            'Engine Size (cc)', 'Fuel Type', 'Insurance', 'Market Specifications',
            'Mileage', 'Model', 'Model Year', 'Paint', 'Transmission'
        ]
        missing = [f for f in required if f not in data or data[f] == '']
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Preprocess
        x = preprocess_car_input(data)

        # Predict (model outputs log1p price → inverse with expm1)
        log_price = model.predict(x)[0]
        price = float(np.expm1(log_price))

        # Price band (±10%)
        low  = round(price * 0.90)
        high = round(price * 1.10)
        price = round(price)

        # Simple recommendation
        if price < 5000:
            label = "Budget-friendly"
        elif price < 15000:
            label = "Good market value"
        elif price < 30000:
            label = "Above average pricing"
        else:
            label = "Premium segment"

        return jsonify({
            'price': price,
            'price_low': low,
            'price_high': high,
            'label': label,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
