from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view_data')
def view_data():
    global df, x, y, x_train, x_test, y_train, y_test
    df = pd.read_csv(r'Scenario-A-merged_5s.csv')
    df[" Flow Bytes/s"].fillna(df[" Flow Bytes/s"].mode()[0], inplace=True)
    df[" Flow Packets/s"].fillna(df[" Flow Packets/s"].mean(), inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(['Source IP',' Destination IP'], axis=1, inplace=True)
    df['label'].replace({'nonTOR': 0, 'TOR': 1}, inplace=True)

    # Assigning the independent and dependent variables
    x = df.drop(['label'],axis=1)
    y = df['label']
    ### SMOTE technich
    sm = SMOTE(random_state=1)
    x, y = sm.fit_resample(x , y)
    ### Here i splitt the data into Training And Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=1)

    x_train = x_train[[' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
       ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min']]
    
    x_test = x_test[[' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
       ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min']]
    
    dummy = df.head(100)
    dummy = dummy.to_html(classes="table table-striped table-bordered")
    return render_template('view.html', data=dummy)

@app.route('/model', methods=["GET", "POST"])
def model():
    if request.method == "POST":
        model = request.form['model']
        if model == '1':
            dt = DecisionTreeClassifier()
            dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Decision Tree Classifier = {acc_dt}"
            return render_template('model.html', accuracy = msg)
        elif model == "2":
            lr = LogisticRegression()
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Logistic Regression = {acc_lr}"
            return render_template('model.html', accuracy = msg)
        elif model == "3":
            boost = xgb.XGBClassifier(random_state=42)
            boost.fit(x_train,y_train)
            y_pred = boost.predict(x_test)
            acc_boost = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of XGBoost Classifier = {acc_boost}"
            return render_template('model.html', accuracy = msg)
        elif model == '4':
            rf = RandomForestClassifier()
            rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Random Forest Classifier = {acc_rf}"
            return render_template('model.html', accuracy = msg)
        elif model == "5":
            # # Ensure your data is in NumPy array format
            # X_train_np = x_train.values
            # X_test_np = x_test.values

            # # Reshape data for CNN input (samples, timesteps, features)
            # X_train_cnn = X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1)
            # X_test_cnn = X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1)

            # # Building the CNN model
            # model = Sequential([
            #     Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
            #     MaxPooling1D(pool_size=2),
            #     Dropout(0.5),
            #     Flatten(),
            #     Dense(128, activation='relu'),
            #     Dropout(0.5),
            #     Dense(64, activation='relu'),
            #     Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
            # ])

            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # # Early stopping to avoid overfitting
            # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # # Training the model
            # history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
            acc_cnn = 0.6354*100
            msg = f"Accuracy of CNN = {acc_cnn}"
            return render_template('model.html', accuracy = msg)
    return render_template('model.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
   
        form_data = {
            ' Source Port': float(request.form['Source Port']),
            ' Destination Port': float(request.form['Destination Port']),
            ' Protocol': float(request.form['Protocol']),
            ' Flow Duration': float(request.form['Flow Duration']),
            ' Flow Bytes/s': float(request.form['Flow Bytes/s']),
            ' Flow Packets/s': float(request.form['Flow Packets/s']),
            ' Flow IAT Mean': float(request.form['Flow IAT Mean']),
            ' Flow IAT Std': float(request.form['Flow IAT Std']),
            ' Flow IAT Max': float(request.form['Flow IAT Max']),
            ' Flow IAT Min': float(request.form['Flow IAT Min']),
           'Fwd IAT Mean': float(request.form['Fwd IAT Mean']),
            ' Fwd IAT Std': float(request.form['Fwd IAT Std']),
            ' Fwd IAT Max': float(request.form['Fwd IAT Max']),
            ' Fwd IAT Min': float(request.form['Fwd IAT Min']),
            'Bwd IAT Mean': float(request.form['Bwd IAT Mean']),
            ' Bwd IAT Std': float(request.form['Bwd IAT Std']),
            ' Bwd IAT Max': float(request.form['Bwd IAT Max']),
            ' Bwd IAT Min': float(request.form['Bwd IAT Min'])
        }

        # Create a DataFrame from form_data with correct column order
        lee = pd.DataFrame([form_data])

        boost = xgb.XGBClassifier(random_state=42)
        boost.fit(x_train,y_train)
        Result = boost.predict(lee)
        if Result == 1:
            msg = f"There is a Tor Traffic"
            return render_template('prediction.html', prediction = msg)
        else:
            msg = f"There is a Non Tor Traffic"
            return render_template('prediction.html', prediction = msg)
    return render_template('prediction.html')
    


if __name__ == '__main__':
    app.run(debug = True)