import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from google import genai
from dotenv import load_dotenv


load_dotenv()

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3-flash-preview"

NEW_REVIEWS = 500
TOP_K = 5

# IMPORTANT:
# This is a new cache because OpenAI embeddings cannot be reused
# with Gemini embeddings.
CACHE_FILE = "gemini_review_embeddings.parquet"

# Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]

    return df


def embed(texts):
    all_embeddings = []

    BATCH_SIZE = 90
    WAIT_SECONDS = 65

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_number = (i // BATCH_SIZE) + 1
        batch = texts[i:i + BATCH_SIZE]

        print(
            f"Embedding batch {batch_number}/{total_batches} "
            f"({len(batch)} reviews)..."
        )

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )

        batch_embeddings = [
            embedding.values
            for embedding in response.embeddings
        ]

        all_embeddings.extend(batch_embeddings)

        # Wait before the next batch so we don't exceed
        # the Gemini free-tier embedding quota.
        if i + BATCH_SIZE < len(texts):
            print(
                f"Waiting {WAIT_SECONDS} seconds "
                "for embedding quota..."
            )
            time.sleep(WAIT_SECONDS)

    return all_embeddings


@st.cache_data()
def load_reviews():

    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()

    df["embedding"] = embed(
        df["comment"].fillna("").tolist()
    )

    df.to_parquet(CACHE_FILE)

    return df


st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews, "
    f"answering with {CHAT_MODEL}"
)


def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors.
    """

    return np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    )


def find_similar_reviews(question, df):

    question_vector = embed([question])[0]

    scores = []

    for review_vector in df["embedding"]:
        scores.append(
            cosine_similarity(
                question_vector,
                review_vector
            )
        )

    df = df.copy()

    df["score"] = scores

    return df.nlargest(TOP_K, "score")


def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"({row['city']}, "
            f"{row['rating']} stars) "
            f"{row['comment']}\n"
        )

    system_prompt = """
    You are an assistant analyzing Zomato customer reviews.

    Answer ONLY using the customer reviews provided.

    Be concise and factual.

    If the provided reviews do not contain enough information
    to answer the question, say:
    "The provided reviews do not contain enough information."
    """

    user_prompt = f"""
    Question:
    {question}

    Reviews:
    {context}
    """

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
        }
    )

    return response.text


# Load reviews
review_df = load_reviews()


# User question
question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common complaints "
        "about delivery?"
    )
)


if question:

    top_reviews = find_similar_reviews(
        question,
        review_df
    )

    answer = ask_llm(
        question,
        top_reviews
    )

    st.markdown("**Answer:**")

    st.write(answer)

    with st.expander(
        "Reviews used to build this answer"
    ):

        st.dataframe(
            top_reviews[
                ["city", "rating", "comment"]
            ],
            hide_index=True
        )


