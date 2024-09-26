import streamlit as st
from statsbombpy import sb
import pandas as pd
import numpy as np

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

#=======================================================================================================================
# ****
# Shots
# ****
def plot_shots_heatmap(player):
    shots_1 = shots[shots['player'] == player]

    shot_location_1 = shots_1['location'].apply(lambda x: [float(i) for i in x[1:-1].split(",") if i.strip()])
    shot_location_1 = np.array(shot_location_1.tolist())

    # Create the pitch layout
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Plot the KDE heatmap using the correct keyword arguments
    sns.kdeplot(
        x=shot_location_1[:, 0],
        y=shot_location_1[:, 1],
        fill=True,  # Fill the area under the density curve
        alpha=0.5,  # Transparency level
        thresh=0.05,  # Threshold for plotting KDE; lower values make the plot fade out more
        n_levels=25,  # Number of contour levels, more gives a smoother gradient
        cmap='RdYlGn_r',  # Color map for the heatmap (change as needed)
        bw_adjust=0.3,  # Bandwidth adjustment for kernel density estimate
        ax=ax
    )#.set_title(f'Shooting Heatmap of {player}')

    return plt


def plot_goals_heatmap(player):
    goals_1 = goals[goals['player'] == player]

    goal_location = goals['location'].apply(lambda x: [float(i) for i in x[1:-1].split(",") if i.strip()])
    goal_location = np.array(goal_location.tolist())

    # Create the pitch layout
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Plot the KDE heatmap using the correct keyword arguments
    sns.kdeplot(
        x=goal_location[:, 0],
        y=goal_location[:, 1],
        fill=True,  # Fill the area under the density curve
        alpha=0.5,  # Transparency level
        thresh=0.05,  # Threshold for plotting KDE; lower values make the plot fade out more
        n_levels=25,  # Number of contour levels, more gives a smoother gradient
        cmap='RdYlGn_r',  # Color map for the heatmap (change as needed)
        bw_adjust=0.3,  # Bandwidth adjustment for kernel density estimate
        ax=ax
    )#.set_title(f'Goals Heatmap of {player}')

    return plt

st.write('# Past Match Performances')

st.header('Shooting Visualization', divider='red')
ids = pd.read_csv('event_ids.csv')

#match_id_shot = st.selectbox('Match ID', ids, key='match_id_shot')
countries = {'England': 'Premier League', 'Spain': 'La Liga', 'Germany': '1. Bundesliga', 'Italy': 'Serie A', 'France': 'Ligue 1',}
seasons = ['2015/2016']
#genders = ['male', 'female']

#events = sb.events(match_id=match_id_shot)
#team_name_shot = st.selectbox('Team', events.team.unique(), key='team_name_shot')
#shots = events[events.type == 'Shot']
country = st.selectbox('Country', countries.keys())
season = st.selectbox('Season', seasons)
#gender = st.selectbox('Gender', genders)

shots = pd.read_csv(f'competition_{country}.csv')

goals = shots[shots['shot_outcome'] == 'Goal']

player_goals = goals['player'].value_counts().reset_index()
player_goals.columns = ['player', 'goals']

#
top_20_players = player_goals.sort_values(by='goals', ascending=False).head(20)

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_20_players['player'], top_20_players['goals'], color='skyblue')
plt.title('Top 20 Goal Scorers')
plt.xlabel('Player')
plt.ylabel('Goals')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
#plt.show()
st.pyplot(plt)


#====
# Goal conversion rates
player_shots = shots['player'].value_counts().reset_index()
player_shots.columns = ['player', 'shots']
top_players = pd.merge(top_20_players, player_shots, on='player')
top_players['goal_conversion_rate'] = top_players['goals'] / top_players['shots']
top_players = top_players.sort_values(by='goal_conversion_rate', ascending=False)
top_players = top_players.reset_index(drop=True)
st.table(top_players)

plt.figure(figsize=(10, 6))
plt.bar(top_players['player'], top_players['goal_conversion_rate'], color='skyblue')
plt.title('Goal Conversion Rate of Top Shots Players')
plt.xlabel('Player')
plt.ylabel('Goal Conversion Rate')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
st.pyplot(plt)

#===
# heatmaps

player_name_shot = st.selectbox('Player', shots.player.unique(), key='player_name_shot')

col1, col2 = st.columns(2)

with col1:
    st.subheader(f'Shooting Heatmap of {player_name_shot}')
    shots_plt = plot_shots_heatmap(player_name_shot)
    st.pyplot(shots_plt)

with col2:
    st.subheader(f'Goals Heatmap of {player_name_shot}')
    goals_plt = plot_goals_heatmap(player_name_shot)
    st.pyplot(goals_plt)


#========================================================================================================================
# ****
# Passing
# ****
# Load event data for a specific match
# You will need to know the match_id for the match you are interested in
st.header('Passing Visualization', divider='red')

match_id_pass = st.selectbox('Match ID', ids, key='match_id_pass')

events = sb.events(match_id=match_id_pass)

# Filter events to get dribbles by a specific player
# Replace 'Player Name' with the name of the player you're interested in
team_name_pass = st.selectbox('Team', events.team.unique(), key='team_name_pass')
passes = events[(events.type == 'Pass') & (events.team == team_name_pass)]
# Extracting End Locations
# End locations are typically stored in a list within the 'location' column or a similar structure
passes['end_location_x'] = passes['location'].apply(lambda x: x[0] if x else None)
passes['end_location_y'] = passes['location'].apply(lambda x: x[1] if x else None)
passes['pass_end_x'] = np.array(passes['pass_end_location'].tolist())[:,0]
passes['pass_end_y'] = np.array(passes['pass_end_location'].tolist())[:,1]
#print(passes.columns)
player_name_pass = st.selectbox('Player', passes.player.unique(), key='player_name_pass')

# Now you have a DataFrame 'passes' with end locations of each pass event
#print(passes[passes['player'] == player][['player', 'end_location_x', 'end_location_y']])
#passes['player'] == 'Lionel Andrés Messi Cuccittini'

player_passes = passes[passes['player'] == player_name_pass][['player', 'end_location_x', 'end_location_y']]

# Create the pitch layout
pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
fig, ax = pitch.draw(figsize=(10, 7))

# Plot the KDE heatmap using the correct keyword arguments
sns.kdeplot(
    x=player_passes['end_location_x'],
    y=player_passes['end_location_y'],
    fill=True,  # Fill the area under the density curve
    alpha=0.5,  # Transparency level
    thresh=0.05,  # Threshold for plotting KDE; lower values make the plot fade out more
    n_levels=25,  # Number of contour levels, more gives a smoother gradient
    cmap='RdYlGn_r',  # Color map for the heatmap (change as needed)
    bw_adjust=0.3,  # Bandwidth adjustment for kernel density estimate
    ax=ax
)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Passing Heatmaps')
    st.pyplot(plt)

#======================================================================================================================

plt.rcParams['axes.unicode_minus'] = False  # Resolving negative sign display issues

# Full name of the player
# In the event data, each player is identified by their full name
# If you don't know the full name, you can retrieve all player names from the dataset metadata
#player_name = 'Roque Mesa Quevedo'

# Start drawing the pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='#52ba7f', line_color='white')
fig, ax = pitch.draw(figsize=(10, 7))

# Filter out events for the specified player and exclude throw-ins
# The 'filter' method can be cascaded for multiple operations
#df_pass = pass_data.filter(
#    lambda e: e.player.name == player_name and e.raw_event['play_pattern'] != 'From Throw In'
#).to_df()  # Convert to a pandas DataFrame
df_pass = passes[passes['player'] == player_name_pass]
df_pass = df_pass[passes['play_pattern'] != 'From Throw In']
#print(df_pass[['location' , 'end_location_x', 'end_location_y']])
#print(df_pass.pass_end_location.to_list())
# Draw arrows to represent passes based on start and end coordinates
pitch.arrows(
    df_pass.end_location_x, df_pass.end_location_y,
    df_pass.pass_end_x, df_pass.pass_end_y,
    color="blue", ax=ax
)
# Draw circles to indicate player positions
pitch.scatter(
    df_pass.end_location_x,
    df_pass.end_location_y,
    alpha=0.2, s=500, color="blue", ax=ax
)
#fig.suptitle('Passes made by %s during the match against %s' % (player_name, team2.name), fontsize=30)
with col2:
    st.subheader('Passing Points')
    st.pyplot(plt)
#plt.show()

##======================================================================================================================
