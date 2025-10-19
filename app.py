import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Optional: set page config
st.set_page_config(page_title="Olympics Analysis", layout="wide")

# ---- Sidebar ----
st.sidebar.title("Olympics Analysis")
st.sidebar.image(
    "https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png",
    use_container_width=True
)

# Preprocessed dataframe
df = preprocessor.preprocess()

user_menu = st.sidebar.radio(
    "Select an Option",
    ("Medal Tally", "Overall Analysis", "Country-wise Analysis", "Athlete-wise Analysis")
)

# =========================
# Medal Tally
# =========================
if user_menu == "Medal Tally":
    st.header("Medal Tally")
    countries, years = helper.country_year_list(df)
    year = st.sidebar.selectbox("Select Year", years)
    country = st.sidebar.selectbox("Select Country", countries)

    medal_tally = helper.medal_tally(df, year, country)

    title_country = country if country != "Overall" else "All countries"
    title_year = str(year) if year != "Overall" else "all years"
    st.title(f"{title_country} performance in {title_year}")

    st.dataframe(medal_tally, use_container_width=True)

# =========================
# Overall Analysis
# =========================
elif user_menu == "Overall Analysis":
    st.title("Top Stats")
    editions = df["Year"].nunique() - 1
    cities = df["City"].nunique()
    sports = df["Sport"].nunique()
    events = df["Event"].nunique()
    athletes = df["Name"].nunique()
    countries = df["region"].nunique()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Athletes")
        st.title(athletes)
    with col3:
        st.header("Countries")
        st.title(countries)

    # Over-time lines
    st.title("Participating countries over the years")
    countries_over_year = helper.data_over_time(df, "region", "No. of Countries")
    fig = px.line(countries_over_year, x="Editions", y="No. of Countries")
    st.plotly_chart(fig, use_container_width=True)

    st.title("Events over the years")
    events_over_year = helper.data_over_time(df, "Event", "No. of events")
    fig = px.line(events_over_year, x="Editions", y="No. of events")
    st.plotly_chart(fig, use_container_width=True)

    st.title("Athletes over the years")
    athletes_over_year = helper.data_over_time(df, "Name", "No. of Players")
    fig = px.line(athletes_over_year, x="Editions", y="No. of Players")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: events per sport per year
    st.title("No. of Events over time (each Sport)")
    fig_hm, ax = plt.subplots(figsize=(20, 20))
    events_matrix = helper.events_per_sports_per_year(df)  # expects pivot-style DF
    sns.heatmap(events_matrix, annot=True, ax=ax)
    st.pyplot(fig_hm)

    # Most successful athletes (global or by sport)
    st.title("Most successful athletes")
    sport_list = df["Sport"].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, "Overall")
    sport = st.selectbox("Select a sport", sport_list)
    players = helper.most_successful(df, sport)
    st.table(players)

# =========================
# Country-wise Analysis
# =========================
elif user_menu == "Country-wise Analysis":
    st.sidebar.header("Country-wise Analysis")
    country_list = sorted(df["region"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Select a Country", country_list)

    # Medal trend over time
    st.title(f"{selected_country} Medal Tally over the years")
    country_df = helper.yearwise_medal_tally(df, selected_country) # type: ignore
    fig = px.line(country_df, x="Year", y="Medal")
    st.plotly_chart(fig, use_container_width=True)

    # Country event heatmap
    st.title(f"{selected_country} excels in the following sports")
    pt = helper.country_event_heatmap(df, selected_country) # type: ignore
    fig_hm, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(pt, annot=True, ax=ax)
    st.pyplot(fig_hm)

    # Top athletes for the country
    st.title(f"Top 10 athletes of {selected_country}")
    top10_df = helper.most_successful_countrywise(df, selected_country) # type: ignore
    st.table(top10_df)

# =========================
# Athlete-wise Analysis
# =========================
elif user_menu == "Athlete-wise Analysis":
    st.title("Athlete-wise Analysis")

    # Use one row per athlete + region to avoid double counts
    athlete_df = df.drop_duplicates(subset=["Name", "region"])

    # Age distributions
    x1 = athlete_df["Age"].dropna()
    x2 = athlete_df[athlete_df["Medal"] == "Gold"]["Age"].dropna()
    x3 = athlete_df[athlete_df["Medal"] == "Silver"]["Age"].dropna()
    x4 = athlete_df[athlete_df["Medal"] == "Bronze"]["Age"].dropna()

    dist_fig = ff.create_distplot(
        [x1, x2, x3, x4],
        ["Overall Age", "Gold Medalist", "Silver Medalist", "Bronze Medalist"],
        show_hist=False,
        show_rug=False
    )
    dist_fig.update_layout(autosize=False, width=1000, height=600)
    st.subheader("Distribution of Age")
    st.plotly_chart(dist_fig, use_container_width=True)

    # Gold medalist age distributions by sport
    x = []
    labels = []
    famous_sports = [
        "Basketball", "Judo", "Football", "Tug-Of-War", "Athletics", "Swimming",
        "Badminton", "Sailing", "Gymnastics", "Art Competitions", "Handball",
        "Weightlifting", "Wrestling", "Water Polo", "Hockey", "Rowing", "Fencing",
        "Shooting", "Boxing", "Taekwondo", "Cycling", "Diving", "Canoeing",
        "Tennis", "Golf", "Softball", "Archery", "Volleyball",
        "Synchronized Swimming", "Table Tennis", "Baseball", "Rhythmic Gymnastics",
        "Rugby Sevens", "Beach Volleyball", "Triathlon", "Rugby", "Polo", "Ice Hockey"
    ]
    for s in famous_sports:
        temp = athlete_df[athlete_df["Sport"] == s]
        x.append(temp[temp["Medal"] == "Gold"]["Age"].dropna())
        labels.append(s)

    dist_fig2 = ff.create_distplot(x, labels, show_hist=False, show_rug=False)
    dist_fig2.update_layout(autosize=False, width=1000, height=600)
    st.subheader("Distribution of Age w.r.t. Sports (Gold Medalists)")
    st.plotly_chart(dist_fig2, use_container_width=True)

    # Height vs Weight scatter (by sport)
    st.subheader("Height vs Weight")
    sport_list = df["Sport"].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, "Overall")
    selected_sport = st.selectbox("Select a Sport", sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)

    fig_scatter, ax = plt.subplots()
    sns.scatterplot(
        x="Weight", y="Height",
        data=temp_df, hue="Medal", style="Sex", s=60, ax=ax
    )
    st.pyplot(fig_scatter)

    # Men vs Women participation over the years
    st.subheader("Men vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig_line = px.line(final, x="Year", y=["Male", "Female"])
    fig_line.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig_line, use_container_width=True)
