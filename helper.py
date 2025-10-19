# helper.py
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Medal Tally + Lists
# -----------------------------
@st.cache_data
def medal_tally(df: pd.DataFrame, year, country) -> pd.DataFrame:
    """
    Returns medal tally by region (default) or by year (when a specific country is chosen).
    Assumes df already contains numeric 'Gold', 'Silver', 'Bronze' columns (from your preprocessor).
    """
    # Deduplicate unique medal-winning rows
    medal_df = df.drop_duplicates(
        subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal']
    )

    flg = 0  # 0 => group by region; 1 => group by year (for a single country across years)

    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    elif year == 'Overall' and country != 'Overall':
        flg = 1
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'Overall' and country == 'Overall':
        # Coerce year to int to match df dtype
        y = int(year)
        temp_df = medal_df[medal_df['Year'] == y]
    else:
        y = int(year)
        temp_df = medal_df[(medal_df['Year'] == y) & (medal_df['region'] == country)]

    if flg == 0:
        x = (
            temp_df.groupby('region')
            .sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']]
            .sort_values('Gold', ascending=False)
            .reset_index()
        )
    else:
        x = (
            temp_df.groupby('Year')
            .sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']]
            .sort_values('Year')
            .reset_index()
        )

    x['Total'] = x['Gold'] + x['Silver'] + x['Bronze']
    # ensure ints for display
    for c in ['Gold', 'Silver', 'Bronze', 'Total']:
        x[c] = x[c].astype(int)

    return x


@st.cache_data
def country_year_list(df: pd.DataFrame):
    """
    Returns (countries, years) in the order your Streamlit UI expects.
    """
    years = sorted(df['Year'].dropna().unique().tolist())
    years.insert(0, 'Overall')

    countries = sorted(df['region'].dropna().unique().tolist())
    countries.insert(0, 'Overall')

    return countries, years


# -----------------------------
# Over-time & Pivot helpers
# -----------------------------
@st.cache_data
def data_over_time(df: pd.DataFrame, col: str, y_axis: str) -> pd.DataFrame:
    """
    Count unique values of `col` per Year, returned as (Editions, y_axis).
    Example: data_over_time(df, 'region', 'No. of Countries')
    """
    # unique (Year, col) pairs then count rows per Year
    data = (
        df.drop_duplicates(subset=['Year', col])
          .groupby('Year')
          .size()
          .reset_index(name=y_axis)
          .sort_values('Year')
    )
    data.rename(columns={'Year': 'Editions'}, inplace=True)
    return data


@st.cache_data
def events_per_sports_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a Sport x Year matrix with counts of distinct Events.
    """
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    return (
        x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count')
         .fillna(0)
         .astype(int)
    )


# -----------------------------
# "Most successful" variants
# -----------------------------
@st.cache_data
def most_successful(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Top medal-winning athletes overall or within a sport.
    Requires numeric Gold/Silver/Bronze columns (created in preprocess).
    """
    temp_df = df.dropna(subset=['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    agg = (
        temp_df.groupby(['Name', 'Sport', 'region'])
               .sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']]
               .reset_index()
    )
    agg['Total'] = agg['Gold'] + agg['Silver'] + agg['Bronze']
    # Primary sort by Gold, then Silver, then Bronze, then Total
    agg = agg.sort_values(['Gold', 'Silver', 'Bronze', 'Total'], ascending=False)

    if sport != 'Overall':
        agg = agg.drop(columns=['Sport'])

    return agg.head(15).reset_index(drop=True)


@st.cache_data
def most_successful_countrywise(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Top 10 athletes for a given country by medals (Gold > Silver > Bronze > Total).
    """
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]

    agg = (
        temp_df.groupby(['Name', 'Sport'])
               .sum(numeric_only=True)[['Gold', 'Silver', 'Bronze']]
               .reset_index()
    )
    agg['Total'] = agg['Gold'] + agg['Silver'] + agg['Bronze']
    agg = agg.sort_values(['Gold', 'Silver', 'Bronze', 'Total'], ascending=False)

    return agg.head(10).reset_index(drop=True)


# -----------------------------
# Country-wise charts/tables
# -----------------------------
@st.cache_data
def yearwise_medal_tally(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Count medals per Year for a given country.
    """
    temp_df = df.dropna(subset=['Medal']).drop_duplicates(
        subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal']
    )
    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year')['Medal'].count().reset_index(name='Medal')
    return final_df


@st.cache_data
def country_event_heatmap(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Pivot of Sport x Year with counts of medals for a given country.
    """
    temp_df = df.dropna(subset=['Medal']).drop_duplicates(
        subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal']
    )
    new_df = temp_df[temp_df['region'] == country]
    pt = new_df.pivot_table(
        index='Sport', columns='Year', values='Medal', aggfunc='count'
    ).fillna(0)
    return pt


# -----------------------------
# Athlete-wise visuals helpers
# -----------------------------
@st.cache_data
def weight_v_height(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Returns a per-athlete table for scatterplot: Weight vs Height (with Medal, Sex).
    """
    cols = ['Name', 'region', 'Sport', 'Weight', 'Height', 'Medal', 'Sex', 'Year']
    base = df.loc[:, [c for c in cols if c in df.columns]].copy()

    # Deduplicate per (Year, Name, region) so that the same athlete can appear across different years,
    # but not multiple times within a single Games.
    base = base.drop_duplicates(subset=['Year', 'Name', 'region'])

    base['Medal'] = base['Medal'].fillna('No Medal')

    if sport != 'Overall':
        base = base[base['Sport'] == sport]

    # Some datasets have zeros or missing for weight/height; keep them but plotting lib can handle NaNs
    return base


@st.cache_data
def men_vs_women(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a Year-wise count of unique male and female participants.
    Uses (Year, Name, region) uniqueness so athletes can be counted in multiple years,
    but only once per year.
    """
    dedup = df.drop_duplicates(subset=['Year', 'Name', 'region'])
    men = dedup[dedup['Sex'] == 'M'].groupby('Year')['Name'].count().reset_index(name='Male')
    women = dedup[dedup['Sex'] == 'F'].groupby('Year')['Name'].count().reset_index(name='Female')

    final = pd.merge(men, women, on='Year', how='outer').sort_values('Year')
    final[['Male', 'Female']] = final[['Male', 'Female']].fillna(0).astype(int)
    return final
