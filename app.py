import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st


def load_dotenv_file(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


ROOT = Path(__file__).parent
load_dotenv_file(ROOT)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ARTIFACT_DIR = ROOT / "artifacts"


@st.cache_resource
def load_pipeline():
    return joblib.load(ARTIFACT_DIR / "match_pipeline.pkl")


pipeline = load_pipeline()


def build_comparison_table(profile_a: dict, profile_b: dict) -> pd.DataFrame:
    numeric_fields = [
        "app_usage_time_min",
        "swipe_right_ratio",
        "likes_received",
        "mutual_matches",
        "profile_pics_count",
        "bio_length",
        "message_sent_count",
        "last_active_hour",
    ]
    rows = []
    for key in numeric_fields:
        value_a = profile_a[key]
        value_b = profile_b[key]
        diff = value_a - value_b
        rows.append(
            {
                "feature": key,
                "user_a": value_a,
                "user_b": value_b,
                "difference_a_minus_b": round(diff, 3) if isinstance(diff, float) else diff,
            }
        )
    shared_interests = sorted(set(profile_a["interest_tags"]) & set(profile_b["interest_tags"]))
    rows.append(
        {
            "feature": "shared_interests_count",
            "user_a": len(profile_a["interest_tags"]),
            "user_b": len(profile_b["interest_tags"]),
            "difference_a_minus_b": len(shared_interests),
        }
    )
    return pd.DataFrame(rows)


def build_feature_row(profile: dict) -> pd.DataFrame:
    row = profile.copy()
    row["interest_count"] = len(profile["interest_tags"])
    row["engagement_score"] = (
        0.30 * profile["likes_received"]
        + 0.45 * profile["mutual_matches"]
        + 0.25 * profile["message_sent_count"]
    )
    row["activity_score"] = profile["app_usage_time_min"] * profile["swipe_right_ratio"]
    row["profile_completeness"] = min(
        1.0, (profile["profile_pics_count"] / 6.0) * 0.4 + (profile["bio_length"] / 500.0) * 0.6
    )
    row["app_usage_time_label"] = "High" if profile["app_usage_time_min"] >= 90 else "Moderate"
    row["swipe_right_label"] = (
        "Swipe Maniac" if profile["swipe_right_ratio"] >= 0.75 else "Balanced"
    )
    row["emoji_usage_rate"] = 0.3
    return pd.DataFrame([row]).drop(columns=["interest_tags", "swipe_time_of_day"])


def compute_pairwise_compatibility(profile_a: dict, profile_b: dict) -> dict:
    shared_interests = set(profile_a["interest_tags"]) & set(profile_b["interest_tags"])
    interest_union = set(profile_a["interest_tags"]) | set(profile_b["interest_tags"])
    interest_score = (len(shared_interests) / max(len(interest_union), 1)) * 35

    def closeness(a: float, b: float, scale: float) -> float:
        return max(0.0, 1.0 - (abs(a - b) / scale))

    usage_score = closeness(profile_a["app_usage_time_min"], profile_b["app_usage_time_min"], 300) * 15
    swipe_score = closeness(profile_a["swipe_right_ratio"], profile_b["swipe_right_ratio"], 1.0) * 15
    activity_pattern_score = usage_score + swipe_score

    engagement_a = 0.30 * profile_a["likes_received"] + 0.45 * profile_a["mutual_matches"] + 0.25 * profile_a["message_sent_count"]
    engagement_b = 0.30 * profile_b["likes_received"] + 0.45 * profile_b["mutual_matches"] + 0.25 * profile_b["message_sent_count"]
    engagement_score = closeness(engagement_a, engagement_b, 250) * 20

    profile_a_quality = min(1.0, (profile_a["profile_pics_count"] / 6.0) * 0.4 + (profile_a["bio_length"] / 500.0) * 0.6)
    profile_b_quality = min(1.0, (profile_b["profile_pics_count"] / 6.0) * 0.4 + (profile_b["bio_length"] / 500.0) * 0.6)
    profile_quality_score = closeness(profile_a_quality, profile_b_quality, 1.0) * 10

    location_score = 10 if profile_a["location_type"] == profile_b["location_type"] else 4
    orientation_score = 10 if profile_a["sexual_orientation"] == profile_b["sexual_orientation"] else 5

    total = min(
        100.0,
        interest_score + activity_pattern_score + engagement_score + profile_quality_score + location_score + orientation_score,
    )
    return {
        "score": total,
        "confidence": total / 100.0,
        "is_match": total >= 60,
        "shared_interests": sorted(shared_interests),
    }


def get_ai_explanation(profile_a: dict, profile_b: dict, score: float, is_match: bool) -> str:
    if not GROQ_API_KEY:
        return (
            "Set `GROQ_API_KEY` to enable LLM explanations. "
            "Current fallback: higher engagement, complete profile, and balanced swipe behavior "
            "usually increase match quality."
        )

    prompt = f"""
You are a dating product assistant.
Model confidence for good match: {score:.3f}
Prediction: {"Good Match" if is_match else "Not Good Match"}
User A profile:
{profile_a}

User B profile:
{profile_b}

Give a concise explanation in 3 bullet points and 1 practical suggestion.
"""
    models_to_try = [
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "llama-3.1-8b-instant",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                },
                timeout=20,
            )
            if response.status_code >= 400:
                error_body = response.text[:500]
                last_error = f"{response.status_code} from Groq for model '{model_name}': {error_body}"
                continue

            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = str(exc)

    return (
        "AI explanation unavailable. "
        f"Reason: {last_error}. "
        "Fallback: higher engagement, complete profile, and balanced swipe behavior usually increase match quality."
    )


st.set_page_config(page_title="Dating Match Predictor", layout="wide")
st.title("Dating Match Predictor + AI Explainability")
st.caption("Model-driven match prediction from profile behavior and demographics.")

INTEREST_OPTIONS = ["Fitness", "Gaming", "Traveling", "Movies", "Reading", "Music", "Coding", "Yoga", "Art", "Photography"]
GENDER_OPTIONS = ["Male", "Female", "Non-binary", "Transgender", "Genderfluid", "Prefer Not to Say"]
ORIENTATION_OPTIONS = ["Straight", "Gay", "Lesbian", "Bisexual", "Pansexual", "Queer", "Asexual", "Demisexual"]
LOCATION_OPTIONS = ["Urban", "Suburban", "Metro", "Small Town", "Rural", "Remote Area"]
INCOME_OPTIONS = ["Very Low", "Low", "Lower-Middle", "Middle", "Upper-Middle", "High", "Very High"]
EDUCATION_OPTIONS = ["No Formal Education", "High School", "Diploma", "Associate’s", "Bachelor’s", "Master’s", "MBA", "PhD", "Postdoc"]
TIME_OPTIONS = ["After Midnight", "Early Morning", "Morning", "Afternoon", "Evening", "Late Night"]


def profile_form(prefix: str, col) -> dict:
    with col:
        st.markdown(f"### User {prefix.upper()}")
        return {
            "gender": st.selectbox("Gender", GENDER_OPTIONS, key=f"{prefix}_gender"),
            "sexual_orientation": st.selectbox("Sexual Orientation", ORIENTATION_OPTIONS, key=f"{prefix}_orientation"),
            "location_type": st.selectbox("Location Type", LOCATION_OPTIONS, key=f"{prefix}_location"),
            "income_bracket": st.selectbox("Income Bracket", INCOME_OPTIONS, key=f"{prefix}_income"),
            "education_level": st.selectbox("Education Level", EDUCATION_OPTIONS, key=f"{prefix}_education"),
            "interest_tags": st.multiselect("Interests", INTEREST_OPTIONS, default=["Fitness", "Traveling", "Music"], key=f"{prefix}_interests"),
            "app_usage_time_min": st.slider("App Usage Time (minutes/day)", 0, 300, 120, key=f"{prefix}_usage"),
            "swipe_right_ratio": st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5, key=f"{prefix}_swipe"),
            "likes_received": st.number_input("Likes Received", 0, 250, 80, key=f"{prefix}_likes"),
            "mutual_matches": st.number_input("Mutual Matches", 0, 60, 15, key=f"{prefix}_matches"),
            "profile_pics_count": st.slider("Profile Pictures", 0, 6, 3, key=f"{prefix}_pics"),
            "bio_length": st.slider("Bio Length", 0, 500, 180, key=f"{prefix}_bio"),
            "message_sent_count": st.number_input("Messages Sent", 0, 500, 70, key=f"{prefix}_messages"),
            "last_active_hour": st.slider("Last Active Hour (0-23)", 0, 23, 20, key=f"{prefix}_active"),
            "swipe_time_of_day": st.selectbox("Swipe Time of Day", TIME_OPTIONS, key=f"{prefix}_time"),
        }


col_a, col_b = st.columns(2)
profile_a = profile_form("a", col_a)
profile_b = profile_form("b", col_b)

if st.button("Predict Match Quality"):
    score_a = float(pipeline.predict_proba(build_feature_row(profile_a))[0, 1])
    score_b = float(pipeline.predict_proba(build_feature_row(profile_b))[0, 1])
    pair_result = compute_pairwise_compatibility(profile_a, profile_b)
    ml_signal = (score_a + score_b) / 2.0
    confidence = (0.75 * pair_result["confidence"]) + (0.25 * ml_signal)
    pred = 1 if confidence >= 0.6 else 0

    st.subheader(f"Prediction: {'Good Match' if pred == 1 else 'Not Good Match'}")
    st.metric("Confidence", f"{confidence:.2%}")
    st.write(f"Pairwise compatibility score: **{pair_result['score']:.1f}/100**")
    st.write(f"Shared interests: **{', '.join(pair_result['shared_interests']) if pair_result['shared_interests'] else 'None'}**")
    st.write(f"User A individual score: **{score_a:.2%}**")
    st.write(f"User B individual score: **{score_b:.2%}**")

    st.subheader("AI Explanation")
    st.info(get_ai_explanation(profile_a, profile_b, confidence, pred == 1))
    st.subheader("Comparison Table (User A vs User B)")
    st.table(build_comparison_table(profile_a, profile_b))