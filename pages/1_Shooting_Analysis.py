from statsbombpy import sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import joblib
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import yaml

import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
from mplsoccer import Pitch


import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
from footer import footer
footer()

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # Since the model is already trained, we just return self.
        # If you later decide you want to allow retraining or further training,
        # you can implement that logic here.
        return self

    def predict_proba(self, X):
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Convert X to a torch tensor
        X_numpy = X.values if isinstance(X, pd.DataFrame) else X
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(X_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Return probabilities as a NumPy array
        return probabilities.numpy()

    def predict(self, X):
        # Get probabilities
        proba = self.predict_proba(X)
        # Convert probabilities to predictions
        return torch.argmax(torch.tensor(proba), dim=1).numpy()

# Define the two hidden layer neural network
class TwoHiddenLayerDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.2):
        super(TwoHiddenLayerDropout, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first linear transformation
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second linear transformation
        self.linear3 = nn.Linear(hidden_dim2, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = self.dropout1(x)  # Apply dropout after the first activation
        x = self.linear2(x)
        x = F.sigmoid(x)
        x = self.dropout2(x)  # Apply dropout after the second activation
        x = self.linear3(x)
        return x

class CNNwithDropout(nn.Module):
    def __init__(self):
        super(CNNwithDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 30 * 20, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 30 * 20)
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.sigmoid(self.fc2(x))
        return x

st.write('# Shooting Analysis')

def get_features(df_shot):
    X = df_shot.drop(['shot_statsbomb_xg',
                      'player',
                      'team'],
                     axis=1, )
    X = X.drop(['id', 'index', 'timestamp', 'second', 'possession_team', 'location', 'match_id', 'shot_end_location',
                'shot_end_x', 'shot_end_y'
                   , 'shot_end_z', 'inside_18_width', 'inside_18_depth', 'type', 'related_events', 'shot_key_pass_id',
                'team_id',
                'player_id', 'possession_team_id', 'shot_freeze_frame', 'possession', 'shot_outcome',
                'shot_saved_off_target',
                'shot_saved_to_post', 'shot_deflected'],
               axis=1, )
    X = X.drop('goal', axis=1)

    le = LabelEncoder()

    boolean_variables = ['under_pressure', 'shot_first_time', 'shot_one_on_one', 'shot_aerial_won', 'shot_open_goal',
                         'out', 'shot_redirect', 'off_camera', 'shot_follows_dribble', 'shot_with_pass', 'inside_18']

    le_X = X[boolean_variables]

    for i in boolean_variables:
        le_X[i] = le.fit_transform(le_X[i])

    le_X = pd.DataFrame(le_X)

    categorical_variables = ['play_pattern', 'position', 'shot_technique', 'shot_body_part', 'shot_type',
                             'assist', 'assist_play_pattern', 'bodypart_angle']

    ohe = OneHotEncoder(categories='auto',
                        handle_unknown='ignore', sparse=False)
    ohe_X_features = ohe.fit_transform(X[categorical_variables])
    ohe_X_labels = ohe.get_feature_names_out()
    ohe_X = pd.DataFrame(ohe_X_features,
                         columns=ohe_X_labels)

    numerical_variables = ['period', 'minute', 'duration', 'shot_distance', 'shot_angle']
    ss = StandardScaler()
    ct = ColumnTransformer([('ss', ss, numerical_variables)])
    ss_X = ct.fit_transform(X)

    ss_X = pd.DataFrame(ss_X,
                        columns=numerical_variables)

    le_X.reset_index(drop=True,
                     inplace=True)
    ohe_X.reset_index(drop=True,
                      inplace=True)
    ss_X.reset_index(drop=True,
                     inplace=True)
    X = pd.concat([le_X, ohe_X, ss_X], axis=1)

    cols = list(json.load(open('features_shot.json')))

    missing_cols = ['assist_Ground Pass', 'assist_High Pass', 'assist_Low Pass',
                    'assist_play_pattern_From Corner', 'assist_play_pattern_From Counter',
                    'assist_play_pattern_From Free Kick',
                    'assist_play_pattern_From Goal Kick', 'assist_play_pattern_From Keeper',
                    'assist_play_pattern_From Kick Off',
                    'assist_play_pattern_From Throw In', 'assist_play_pattern_Other',
                    'assist_play_pattern_Regular Play']
    nan_cols = ['assist_nan', 'assist_play_pattern_nan']
    lst = []
    for col in missing_cols + nan_cols:
        if col not in X.columns:
            lst.append(True)
    if all(lst):
        X[missing_cols] = 0
        X[nan_cols] = 1

    for col in cols:
        if col not in X.columns:
            X[col] = 0

    X = X[cols]

    return X

@st.cache_data
def cnn_data_cleaning(shots: pd.DataFrame):
    shot_b_cols = ['under_pressure', 'shot_first_time', 'shot_one_on_one', 'shot_aerial_won', 'shot_open_goal',
                   'shot_saved_off_target', 'out', 'shot_saved_to_post',
                   'shot_redirect', 'shot_deflected', 'off_camera', 'shot_follows_dribble']

    shots[shot_b_cols] = shots[shot_b_cols].replace('TRUE', 1).fillna(0).astype(bool)
    # set binary classification labels
    shots['goal'] = shots['shot_outcome'].apply(lambda x: True if x == 'Goal' else False)

    shots['shot_with_pass'] = shots['shot_key_pass_id'].apply(lambda x: True if x != np.nan else False)

    shots['shot_key_pass_id'] = shots['shot_key_pass_id'].fillna('No Pass')
    shots['location'] = shots['location'].apply(lambda x: [float(i) for i in x[1:-1].split(",") if i.strip()])
    shots['player_x'] = np.array(shots['location'].tolist())[:, 0]
    shots['player_y'] = np.array(shots['location'].tolist())[:, 1]
    shots['shot_end_x'] = shots['shot_end_location'].apply(lambda x: x[0])
    shots['shot_end_y'] = shots['shot_end_location'].apply(lambda x: x[1])
    shots['shot_end_z'] = shots['shot_end_location'].apply(lambda x: x[2] if len(x) == 3 else -1)

    idx = []
    for i, s in shots.iterrows():
        if s['player_x'] > 120:
            s['player_x'] = 120
        if s['player_y'] > 80:
            s['player_y'] = 80
    # drop balls out of court
    #shots.drop(index=idx, inplace=True)

    freeze_frames = []

    shots['shot_freeze_frame'] = shots['shot_freeze_frame'].apply(lambda x: x.replace('True', '1'))
    shots['shot_freeze_frame'] = shots['shot_freeze_frame'].apply(lambda x: x.replace('False', '0'))
    shots['shot_freeze_frame'] = shots['shot_freeze_frame'].apply(lambda x: yaml.safe_load(x.replace("'", '"')))
    for freeze_frame in shots['shot_freeze_frame']:
        if isinstance(freeze_frame, float):
            freeze_frames.append([])
        else:
            freeze_frames.append(freeze_frame)

    team_heatmaps = []
    opponent_heatmaps = []

    # for each frame
    for freeze_frame in freeze_frames:
        curr_team_heatmap = [[0, ] * 80 for _ in range(120)]
        curr_opponent_heatmap = [[0, ] * 80 for _ in range(120)]

        team, opponent = 0, 0

        # for each player
        for free_frame_record in freeze_frame:
            i = -1 if free_frame_record['location'][0] >= 120 else int(free_frame_record['location'][0])
            j = -1 if free_frame_record['location'][1] >= 80 else int(free_frame_record['location'][1])

            # count player's heatmap
            if free_frame_record['teammate']:
                curr_team_heatmap[i][j] += 1
                team += 1
            else:
                curr_opponent_heatmap[  i][j] += 1
                opponent += 1

        team_heatmaps.append(curr_team_heatmap)
        opponent_heatmaps.append(curr_opponent_heatmap)

    ball_heatmaps = []

    for location in shots['location']:
        ball_heatmap = [[0, ] * 80 for _ in range(120)]

        i = -1 if location[0] >= 120 else int(location[0])
        j = -1 if location[1] >= 80 else int(location[1])

        ball_heatmap[i][j] += 1

        ball_heatmaps.append(ball_heatmap)

    team_heatmaps = torch.tensor(team_heatmaps, dtype=torch.float32)
    opponent_heatmaps = torch.tensor(opponent_heatmaps, dtype=torch.float32)
    ball_heatmaps = torch.tensor(ball_heatmaps, dtype=torch.float32)
    X_ = torch.stack((team_heatmaps, opponent_heatmaps, ball_heatmaps), dim=1)
    #y_ = torch.tensor(shots['goal'], dtype=torch.float32)

    return X_


df_shot = st.file_uploader("Choose a file", type='csv')
if df_shot is not None:
    df_shot = pd.read_csv(df_shot)
    try:
        X = get_features(df_shot)
        X_frz_pn = cnn_data_cleaning(df_shot)
    except:
        st.warning("Invalid file format. Please upload a Statsbomb formated shots file.")
        sys.exit()
    #st.write(X_frz_pn)

    vc = joblib.load('vc.pkl')
    cnn = CNNwithDropout()
    cnn.load_state_dict(torch.load('cnn_shot.pth'))
    cnn.eval()
    #st.write(cnn(X_frz_pn).T)
    vc_probs = vc.predict_proba(X)[:, 1]
    cnn_probs = cnn(X_frz_pn).T[0].cpu().detach().numpy()
    predictions = (vc_probs + cnn_probs) / 2
    #predictions = vc_probs
    #st.write(predictions)
    #X['expected_goals'] = predictions

    #================================
    #plot
    df_shot['expected_goals'] = predictions


    st.subheader('Expected Goals', divider='red')

    team = st.selectbox('Team', np.append(df_shot['team'].unique(), 'All'))
    if team != 'All':
        df_shot = df_shot[df_shot['team'] == team]

    st.write(f'#### Team {team} Expected Goals')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    plt.scatter(df_shot['player_x'], df_shot['player_y'], c='red', s=df_shot['expected_goals'] * 1000, edgecolors='black', alpha=0.5)
    if len(df_shot) < 50:
        for i, txt in enumerate(df_shot['expected_goals']):
            ax.annotate(round(txt, 2), (df_shot['player_x'].iloc[i] - 2, df_shot['player_y'].iloc[i] - 3))
    st.pyplot(plt)


    shot_counts = df_shot.groupby('player').count().sort_values('id', ascending=False)['id']
    shot_counts = shot_counts.reset_index().rename(columns={'id': 'shot_count'})

    st.write(f'#### Top Shooters')
    plt.figure(figsize=(10, 6))
    plt.bar(shot_counts['player'][:20], shot_counts['shot_count'][:20], color='skyblue')
    plt.xlabel('Player')
    plt.ylabel('Shots')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    st.pyplot(plt)
    #st.write(df_shot)

    st.write(f'#### Top Average Expected Goals')
    xg_avg = df_shot[['player', 'expected_goals']].groupby('player').mean().sort_values('expected_goals', ascending=False)
    xg_avg = xg_avg.reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(xg_avg['player'][:20], xg_avg['expected_goals'][:20], color='skyblue')
    plt.xlabel('Player')
    plt.ylabel('Expected Goals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.ylim([0, 1])
    st.pyplot(plt)



    st.subheader('Player Expected Goals', divider='red')
    df_shot['time'] = df_shot['minute'].astype('str') + [':'] + df_shot['second'].astype('str')
    player_name = st.selectbox('Player', df_shot['player'].unique())
    player = df_shot.loc[df_shot['player'] == player_name]

    st.write(f'#### {player_name} Expected Goals')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))
    plt.scatter(player['player_x'], player['player_y'], c='red', s=player['expected_goals'] * 1000, edgecolors='black', alpha=0.5)
    for i, txt in enumerate(player['expected_goals']):
        ax.annotate(round(txt, 2), (player['player_x'].iloc[i] - 2, player['player_y'].iloc[i] - 3))
    st.pyplot(plt)

    st.write(f'#### {player_name} Match Expected Goals Performance Trend')
    plt.figure(figsize=(10, 6))
    plt.plot(player['time'], player['expected_goals'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Expected Goals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.ylim([0, 1])
    st.pyplot(plt)