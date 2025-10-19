import pandas as pd
import streamlit as st

@st.cache_data
def preprocess():
    # load datasets (adjust paths if needed)
    df = pd.read_csv('data/athlete_events.csv')
    region_df = pd.read_csv('data/noc_regions.csv')

    # keep Summer only
    df = df[df['Season'] == 'Summer']

    # add region info
    df = df.merge(region_df, on="NOC", how='left')

    # remove perfect duplicates
    df = df.drop_duplicates()

    # one-hot encode Medal into Bronze/Gold/Silver (0/1)
    medals = pd.get_dummies(df['Medal'], dtype=int)

    # ensure all three columns exist, even if a medal type is absent in a filter
    for col in ['Gold', 'Silver', 'Bronze']:
        if col not in medals.columns:
            medals[col] = 0
    medals = medals[['Gold', 'Silver', 'Bronze']]

    # attach to df
    df = pd.concat([df, medals], axis=1)

    return df
