import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import re

# --- Title ---
st.title("CORD-19 Metadata Analysis")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("metadata_small.csv.gz")
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df.dropna(subset=['publish_time'], inplace=True)
    return df

df = load_data()
st.write("Dataset Preview:", df.head())

# --- Time Series Plot ---
papers_by_date = df.groupby(pd.Grouper(key='publish_time', freq='M')).size()
st.subheader("Papers Published Over Time")
st.line_chart(papers_by_date)

# --- Top Journals ---
df['journal'] = df['journal'].replace(r'^\s*$', pd.NA, regex=True)

# Count
top_journals = df['journal'].value_counts().head(10)
st.subheader("Top Journals")

# Plot top 10 journals
if not top_journals.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    top_journals.plot(kind='bar', ax=ax, title="Top 10 Journals Publishing Covid-19 Papers")
    ax.set_xlabel("Journal")
    ax.set_ylabel("Number of Papers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("No journal data available to plot.")


# --- Most frequent words in title

def plot_title_analysis(df):
    st.subheader("Most Frequent Words Used in Paper Titles")

    # Option selector
    option = st.radio(
        "Choose visualization:",
        ("Bar Chart (Top Words)", "Word Cloud"),
        horizontal=True
    )

    # --- Preprocess titles ---
    all_titles = " ".join(df['title'].dropna().astype(str).tolist()).lower()

    stopwords = set(STOPWORDS)
    stopwords.update([
        "using","based","study","analysis","of","in","and","the","for","with","to",
        "on","by","from","an","a","at","is","are","be","as","that","this","it",
        "its","into","was","were","or","abstract","expression","following","high",
        "type","approach","use","method"
    ])

    if option == "Bar Chart (Top Words)":
        # Slider for number of words
        top_n = st.slider("Select number of top words to display:", 5, 50, 20)

        words = re.findall(r'\b[a-z]{2,}\b', all_titles)
        filtered_words = [w for w in words if w not in stopwords]
        word_freq = Counter(filtered_words).most_common(top_n)

        if word_freq:
            words, counts = zip(*word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(words, counts)
            ax.set_title(f"Top {top_n} Most Frequent Words in Titles")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No words extracted. Check if 'title' column has values.")

    elif option == "Word Cloud":
        if not all_titles.strip():
            st.warning("No titles available to generate word cloud.")
            return

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            stopwords=stopwords
        ).generate(all_titles)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud of Paper Titles (Filtered Stopwords)", fontsize=16)
        st.pyplot(fig)

plot_title_analysis(df)

import matplotlib.pyplot as plt
import streamlit as st

def plot_journal_longtail(df):
    st.subheader("Long-tail Distribution of Journals")

    # Count papers per journal
    journal_counts = df['journal'].value_counts()

    if journal_counts.empty:
        st.warning("No journal data available.")
        return

    # Long-tail distribution (log-log plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    journal_counts.reset_index(drop=True).plot(
        logy=True, logx=True, marker='o', linestyle='none', ax=ax
    )
    ax.set_title("Long-tail Distribution of Journals")
    ax.set_xlabel("Rank of Journal")
    ax.set_ylabel("Number of Papers (log scale)")
    st.pyplot(fig)

plot_journal_longtail(df)
