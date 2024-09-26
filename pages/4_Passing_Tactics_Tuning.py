import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import display
import streamlit as st
import joblib
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
from footer import footer
footer()
st.write('# Passing Tactics Tuning')

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
# Set a random seed for both CPU and GPU
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # if using multi-GPU.

# For torch models, we need to define a function to calculate the metrics
# Calculate metrics for binary classification (using Cross Entropy Loss)
def calculate_metrics_with_CEL(model, x, y):
    model.eval()
    with torch.no_grad():
        y_prob = model(x)
        y_pred = torch.argmax(y_prob, dim=1)
        acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy()).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print(f"Accuracy: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    return acc, tpr, fpr

# Calculate metrics for binary classification (using Binary Cross Entropy)
def calculate_metrics_with_BCE(model,x,y):
    model.eval()
    with torch.no_grad():
        y_prob = model(x)
        y_pred = (y_prob > 0.5).float()
        acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy()).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print(f"Accuracy: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    return acc, tpr, fpr

# Define similar function for statmodels
def calculate_metrics_no_torch(model, x, y):
    y_prob = model.predict(x)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print(f"Accuracy: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    return acc, tpr, fpr



import time

def train_model(model, loss_func, num_epochs, optimizer, train_loader, x_test, y_test):

  train_loss_log = []
  test_loss_log = []

  # Move model to GPU if CUDA is available
  if torch.cuda.is_available():
      model = model.cuda()
      x_test = x_test.cuda()
      y_test = y_test.cuda()
  tic = time.time()
  for epoch in range(1,num_epochs+1):
    for i, data in enumerate(train_loader):
      x, y = data
      # check if cuda is available
      if torch.cuda.is_available():
        x , y = x.cuda(), y.cuda()
      # get predicted y value from our current model
      pred_y = model(x)
      # calculate the loss
      loss = loss_func(pred_y,y)
      # Zero the gradient of the optimizer
      optimizer.zero_grad()
      # Backward pass: Compute gradient of the loss with respect to model parameters
      loss.backward()
      # update weights
      optimizer.step()
    # change the model to evaluation mode to calculate the test loss; We will come back to this later after learning Dropout and Batch Normalization
    train_loss_log.append(loss.item())
    model.eval()
    test_pred_y = model(x_test)
    test_loss = loss_func(test_pred_y,y_test)
    test_loss_log.append(test_loss.item())
    # change back to training mode.
    model.train()
    print("Epoch {:2},  Training Loss: {:9.4f},  Test Loss: {:7.4f}".format(epoch, loss.item(), test_loss.item()))
  toc = time.time()
  print("Elapsed Time : {:7.2f}".format(toc-tic))
  return train_loss_log, test_loss_log

# define single hidden layer neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

# define single hidden layer neural network
class SingleHiddenLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SingleHiddenLayerNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2) # If we are using CrossEntropyLoss, we need to have 2 output nodes for binary classification

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the DataFrame has two columns: features and target
        x = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx,0], dtype=torch.long) # return 1 dimensional tensor
        return x, y



cnn_model = joblib.load('cnn_model.joblib')
rf = joblib.load('rf_model_1.pkl')
xgb = joblib.load('xgboost_pass.pkl')

# Initialize session state for storing attempts and latest prediction
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Attempt', 'Probability', 'Input Details'])
if 'latest_prediction' not in st.session_state:
    st.session_state['latest_prediction'] = 0

# Function to clear the plot data
def clear_data():
    st.session_state['data'] = pd.DataFrame(columns=['Attempt', 'Probability', 'Input Details'])

def add_data_point():
    predicted_probability = st.session_state['latest_prediction']
    input_details = (f"Player X: {player_x},<br>Player Y: {player_y},<br>Pass End X: {pass_end_x},<br>"
                     f"Pass End Y: {pass_end_y},<br>Pass Length: {pass_length},<br>Pass Angle: {pass_angle},<br>"
                     f"Pass Switch: {pass_switch},<br>Under Pressure: {under_pressure},<br>"
                     f"Pass Shot Assist: {pass_shot_assist},<br>Pass Cross: {pass_cross},<br>"
                     f"Pass Cut Back: {pass_cut_back},<br>Counterpress: {counterpress},<br>"
                     f"Pass Through Ball: {pass_through_ball},<br>Pass Deflected: {pass_deflected},<br>"
                     f"Pass Inswinging: {pass_inswinging},<br>Pass Outswinging: {pass_outswinging},<br>"
                     f"Pass Goal Assist: {pass_goal_assist},<br>Pass Straight: {pass_straight},<br>"
                     f"Pass No Touch: {pass_no_touch},<br>")
    new_attempt = len(st.session_state['data']) + 1
    st.session_state['data'] = pd.concat([st.session_state['data'], pd.DataFrame({'Attempt': [new_attempt], 'Probability': [predicted_probability], 'Input Details': [input_details]})], ignore_index=True)

def update_prediction():
    columns = ['player_x', 'player_y', 'pass_end_x', 'pass_end_y', 'pass_length', 'pass_angle',
               'pass_switch', 'under_pressure', 'pass_shot_assist', 'pass_cross', 'pass_cut_back',
               'counterpress', 'pass_through_ball', 'pass_deflected', 'pass_inswinging', 'pass_outswinging',
               'pass_goal_assist', 'pass_straight', 'pass_no_touch']
    x = {col: 0 for col in columns}

    # 1. numeric
    x['player_x'] = player_x
    x['player_y'] = player_y
    x['pass_end_x'] = pass_end_x
    x['pass_end_y'] = pass_end_y
    x['pass_length'] = pass_length
    x['pass_angle'] = pass_angle
    # 2. boolean
    boolean_features = {'pass_switch': pass_switch,
                        'under_pressure': under_pressure,
                        'pass_shot_assist': pass_shot_assist,
                        'pass_cross': pass_cross,
                        'pass_cut_back': pass_cut_back,
                        'counterpress': counterpress,
                        'pass_through_ball': pass_through_ball,
                        'pass_deflected': pass_deflected,
                        'pass_inswinging': pass_inswinging,
                        'pass_outswinging': pass_outswinging,
                        'pass_goal_assist': pass_goal_assist,
                        'pass_straight': pass_straight,
                        'pass_no_touch': pass_no_touch}
    for boolean_feature in boolean_features:
        if boolean_features[boolean_feature]:
            x[boolean_feature] = 1

    df = pd.DataFrame(columns=columns)
    df = pd.concat([df, pd.DataFrame([x])], ignore_index=True)

    # Convert the DataFrame to a PyTorch tensor
    tensor = torch.tensor(df.values.astype('float32'))
    cnn_model.eval()
    # Pass the tensor through the model
    cnn_probs = torch.softmax(cnn_model(tensor), dim=1)[:, 1].cpu().detach().numpy()[0]
    rf_probs = rf.predict_proba(df)[0, 1]
    df_xgb = df.copy()
    df_xgb[list(boolean_features.keys())] = df_xgb[list(boolean_features.keys())].astype('bool')
    xgb_probs = xgb.predict_proba(df_xgb)[0, 1]

    predictions = (cnn_probs + rf_probs + xgb_probs) / 3

    st.session_state['latest_prediction'] = predictions
    metrics_display.metric(label="Probability of Passing", value=f"{predictions * 100:.2f}%")


col1, col2, col3 = st.columns(3)

with col1:
    player_x = st.slider('Player X', min_value=0.00, max_value=121.00, step=0.1, value=50.00, key='player_x')
    player_y = st.slider('Player Y', min_value=0.00, max_value=80.00, step=0.1, value=50.00, key='player_y')
with col2:
    pass_end_x = st.slider('Pass End X', min_value=0.00, max_value=121.00, step=0.1, value=50.00, key='pass_end_x')
    pass_end_y = st.slider('Pass End Y', min_value=0.00, max_value=80.00, step=0.1, value=50.00, key='pass_end_y')
with col3:
    pass_length = st.slider('Pass Length', min_value=0.00, max_value=122.00, step=0.1, value=50.00, key='pass_length')
    pass_angle = st.slider('Pass Angle', min_value=-3.15, max_value=3.15, step=0.01, value=1.00, key='pass_angle') # in radians (-pipi)
pass_switch = st.checkbox('Pass Switch', value=False, key='pass_switch')
under_pressure = st.checkbox('Under Pressure', value=False, key='under_pressure')
pass_shot_assist = st.checkbox('Pass Shot Assist', value=False, key='pass_shot_assist')
pass_cross = st.checkbox('Pass Cross', value=False, key='pass_cross')
pass_cut_back = st.checkbox('Pass Cut Back', value=False, key='pass_cut_back')
counterpress = st.checkbox('Counter Press', value=False, key='counterpress')
pass_through_ball = st.checkbox('Pass Through Ball', value=False, key='pass_through_ball')
pass_deflected = st.checkbox('Pass Deflected', value=False, key='pass_deflected')
pass_inswinging = st.checkbox('Pass Inswinging', value=False, key='pass_inswinging')
pass_outswinging = st.checkbox('Pass Outswinging', value=False, key='pass_outswinging')
pass_goal_assist = st.checkbox('Pass Goal Assist', value=False, key='pass_goal_assist')
pass_straight = st.checkbox('Pass Straight', value=False, key='pass_straight')
pass_no_touch = st.checkbox('Pass No Touch', value=False, key='pass_no_touch')


metrics_display = st.empty()

# Place a button to add a data point to the plot
st.button("Add Data Point", on_click=add_data_point)

# Button to clear the plot
st.button("Clear Plot", on_click=clear_data)

# Call update_prediction whenever the input changes to update metrics display in real-time
update_prediction()

# Plotting
fig = go.Figure()
if not st.session_state['data'].empty:
    fig.add_trace(go.Scatter(x=st.session_state['data']['Attempt'], y=st.session_state['data']['Probability'],
                             mode='markers+lines', name='Probability',
                             marker=dict(color=st.session_state['data']['Probability'].apply(lambda x: 'green' if x > 0.5 else 'red'), size=10),
                             text=st.session_state['data']['Input Details'], hoverinfo='text'))
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="Threshold (0.5)", annotation_position="bottom right")
    fig.update_layout(title='Prediction Attempts Over Time', xaxis_title='Attempt', yaxis_title='Predicted Probability',
                      yaxis=dict(range=[0, 1]), hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)