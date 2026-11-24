
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# Load saved model and movie data
@st.cache_resource
def load_model():
    return joblib.load("movie_recommendation_model.joblib")


try:
    model_bundle = load_model()

    best_model = model_bundle["model"]
    genre_classes = model_bundle["genre_classes"]
    movie_data = model_bundle["movie_data"]

except Exception as e:
    st.error("Unable to load the model file.")
    st.error(str(e))
    st.stop()


# Check dataset columns
required_columns = [
    "movie_name",
    "genre",
    "genre_list",
    "overview",
    "cast",
    "director"
]

if not all(col in movie_data.columns for col in required_columns):
    st.error("Dataset columns do not match the required format.")
    st.stop()


# Application heading
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies based on your favourite genre "
    "and two favourite actors."
)

st.divider()


# User input section
st.subheader("Enter Your Preferences")

selected_genre = st.selectbox(
    "Select Movie Genre",
    sorted(genre_classes)
)

actor1 = st.text_input(
    "Enter First Actor Name",
    placeholder="Example: Shah Rukh Khan"
)

actor2 = st.text_input(
    "Enter Second Actor Name",
    placeholder="Example: Deepika Padukone"
)

number_of_movies = st.slider(
    "Number of Movies",
    min_value=1,
    max_value=20,
    value=10
)


# Recommendation function
def recommend_movies(genre, actor1, actor2, limit):

    results = movie_data.copy()

    # Remove missing values from important columns
    results["cast"] = results["cast"].fillna("")
    results["genre_list"] = results["genre_list"].apply(
        lambda x: x if isinstance(x, (list, tuple, set)) else []
    )

    # Filter movies by genre
    results = results[
        results["genre_list"].apply(
            lambda genres: any(
                str(g).strip().casefold() == genre.strip().casefold()
                for g in genres
            )
        )
    ]

    # Filter by first actor
    if actor1.strip():

        results = results[
            results["cast"].str.contains(
                actor1.strip(),
                case=False,
                regex=False
            )
        ]

    # Filter by second actor
    if actor2.strip():

        results = results[
            results["cast"].str.contains(
                actor2.strip(),
                case=False,
                regex=False
            )
        ]

    # Remove duplicate movie names
    results = results.drop_duplicates(subset=["movie_name"])

    return results.head(limit)


# Recommendation button
if st.button("Recommend Movies", type="primary"):

    if actor1.strip() and actor2.strip():
        st.info(
            f"Searching for {selected_genre} movies "
            f"featuring both {actor1.strip()} and {actor2.strip()}."
        )

    elif actor1.strip():
        st.info(f"Searching for {selected_genre} movies featuring {actor1.strip()}.")

    elif actor2.strip():
        st.info(f"Searching for {selected_genre} movies featuring {actor2.strip()}.")

    else:
        st.info(f"Searching for {selected_genre} movies.")

    results = recommend_movies(
        selected_genre,
        actor1,
        actor2,
        number_of_movies
    )

    if results.empty:

        st.warning(
            "No matching movies found. "
            "Please try another genre or actor combination."
        )

    else:

        st.success(
            f"{len(results)} matching movies found!"
        )

        st.subheader("Recommended Movies")

        for _, movie in results.iterrows():

            with st.container(border=True):

                st.subheader(str(movie["movie_name"]))

                st.write("**Description:**")

                st.write(
                    movie["overview"]
                    if pd.notna(movie["overview"])
                    else "Description not available."
                )

                st.write(
                    "**Genre:**",
                    movie["genre"]
                    if pd.notna(movie["genre"])
                    else "Not available"
                )

                st.write(
                    "**Cast:**",
                    movie["cast"]
                    if movie["cast"]
                    else "Not available"
                )

                st.write(
                    "**Director:**",
                    movie["director"]
                    if pd.notna(movie["director"])
                    else "Not available"
                )


# Footer
st.divider()

st.caption(
    "Movie Recommendation System | Computational Intelligence Project"
)
