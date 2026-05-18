from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    model = IsolationForest(contamination=0.05)
    return model.fit_predict(data)
