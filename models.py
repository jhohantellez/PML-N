import pickle

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('rf_encoder.pkl', 'rb') as f:
    rf_encoder = pickle.load(f)
with open('rf_le_dept.pkl', 'rb') as f:
    rf_le_dept = pickle.load(f)
with open('rf_le_event.pkl', 'rb') as f:
    rf_le_event = pickle.load(f)