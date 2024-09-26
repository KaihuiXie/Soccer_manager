from statsbombpy import sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import joblib
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch


import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
from footer import footer
footer()

st.write('# Passing Analysis')
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

# Build Datasets and DataLoaders
from torch.utils.data import DataLoader, Dataset

# Start your code here. Refer to the previous labs for examples.

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the DataFrame has two columns: features and target
        x = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx,0], dtype=torch.long)
        return x, y

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


cnn_model = joblib.load('cnn_model.joblib')
rf = joblib.load('rf_model_1.pkl')
xgb = joblib.load('xgboost_pass.pkl')

df_pass = st.file_uploader("Choose a file", type='csv')
if df_pass is not None:
    df_pass = pd.read_csv(df_pass)
    features = [
        'player_x', 'player_y', 'pass_end_x', 'pass_end_y', 'pass_length', 'pass_angle',
        'pass_switch', 'under_pressure', 'pass_shot_assist', 'pass_cross', 'pass_cut_back',
        'counterpress', 'pass_through_ball', 'pass_deflected', 'pass_inswinging', 'pass_outswinging',
        'pass_goal_assist', 'pass_straight', 'pass_no_touch'
    ]
    target = 'pass_outcome'
    allcols = [ 'pass_outcome',
        'player_x', 'player_y', 'pass_end_x', 'pass_end_y', 'pass_length', 'pass_angle',
        'pass_switch', 'under_pressure', 'pass_shot_assist', 'pass_cross', 'pass_cut_back',
        'counterpress', 'pass_through_ball', 'pass_deflected', 'pass_inswinging', 'pass_outswinging',
        'pass_goal_assist', 'pass_straight', 'pass_no_touch'
    ]

    # Prepare the dataset
    try:
        X = df_pass[features]
        y = df_pass[target]
    except:
        st.warning("Invalid file format. Please upload a Statsbomb formated passing file.")
        sys.exit()

    Test = pd.concat([y, X], axis=1)
    Test_dataset = CustomDataset(Test)
    Test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=True)

    X_test_torch = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
    y_test_torch = torch.tensor(y.values, dtype=torch.long)

    cnn_model.eval()
    cnn_probs = torch.softmax(cnn_model(X_test_torch), dim=1)[:, 1].cpu().detach().numpy()
    rf_probs = rf.predict_proba(X)[:, 1]
    xgb_probs = xgb.predict_proba(X)[:, 1]
    #st.write(predictions)
    predictions = (cnn_probs + rf_probs + xgb_probs) / 3



    df_pass['expected_pass'] = predictions

    st.subheader('Expected Pass', divider='red')

    team = st.selectbox('Team', np.append(df_pass['team'].unique(), 'All'))
    if team != 'All':
        df_pass = df_pass[df_pass['team'] == team]

    st.write(f'#### Team {team} Passing Predictions')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    plt.scatter(df_pass['player_x'], df_pass['player_y'], c='red', s=df_pass['expected_pass'] * 1000,
                edgecolors='black', alpha=0.5)
    st.pyplot(plt)

    shot_counts = df_pass.groupby('player').count().sort_values('id', ascending=False)['id']
    shot_counts = shot_counts.reset_index().rename(columns={'id': 'shot_count'})

    st.write('Top Passers')
    plt.figure(figsize=(10, 6))
    plt.bar(shot_counts['player'][:20], shot_counts['shot_count'][:20], color='skyblue')
    plt.xlabel('Player')
    plt.ylabel('Pass')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    st.pyplot(plt)
    # st.write(df_pass)

    st.write('#### Top Average Expected Pass')
    xg_avg = df_pass[['player', 'expected_pass']].groupby('player').mean().sort_values('expected_pass', ascending=False)
    xg_avg = xg_avg.reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(xg_avg['player'][:20], xg_avg['expected_pass'][:20], color='skyblue')
    plt.title('Top Average Expected Pass')
    plt.xlabel('Player')
    plt.ylabel('Expected Pass')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    st.pyplot(plt)

    st.subheader('Player Expected Pass', divider='red')
    df_pass['time'] = df_pass['minute'].astype('str') + [':'] + df_pass['second'].astype('str')
    player_name = st.selectbox('Player', df_pass['player'].unique())
    player = df_pass.loc[df_pass['player'] == player_name]


    st.write(f'#### {player_name} Expected Pass')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.arrows(
        player.player_x, player.player_y,
        player.pass_end_x, player.pass_end_y,
        color="red", alpha=0.5, ax=ax
    )
    pitch.scatter(player['player_x'], player['player_y'], c='red', s=player['expected_pass'] * 1000, edgecolors='black',
                alpha=0.5, ax=ax)
    for i, txt in enumerate(player['expected_pass']):
        ax.annotate(round(txt, 2), (player['player_x'].iloc[i] - 2, player['player_y'].iloc[i] - 3))
    st.pyplot(plt)

    st.write(f'#### {player_name} Match Expected Pass Performance Trend')
    plt.figure(figsize=(10, 6))
    plt.plot(player['time'], player['expected_pass'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Expected Pass')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.ylim([0, 1])
    st.pyplot(plt)
