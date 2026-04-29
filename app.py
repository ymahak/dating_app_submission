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
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(212, 89, 145, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(137, 65, 117, 0.2), transparent 22%),
            linear-gradient(180deg, #141018 0%, #100c14 45%, #0c0910 100%);
        animation: fadeInPage 0.8s ease-in-out;
        color: #f7eef5;
    }
    .hero {
        background: rgba(20, 16, 24, 0.96);
        padding: 2.2rem 2.3rem;
        border-radius: 28px;
        color: #fff4fb;
        border: 1px solid #3f2e3a;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.35rem;
        animation: slideUp 0.7s ease-out;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .hero:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -50px;
        right: -40px;
        width: 220px;
        height: 220px;
        background: rgba(255, 255, 255, 0.14);
        border-radius: 50%;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.55rem;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .hero p {
        margin: 0.8rem 0 0 0;
        opacity: 1;
        line-height: 1.65;
        max-width: 720px;
        font-size: 1.02rem;
        color: #fff4fb;
    }
    .eyebrow {
        display: inline-block;
        padding: 0.36rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.4);
        font-size: 0.82rem;
        letter-spacing: 0.4px;
        margin-bottom: 0.9rem;
        color: #fff4fb;
    }
    .sub-card {
        background: rgba(35, 24, 33, 0.92);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        border: 1px solid #513547;
        box-shadow: 0 10px 26px rgba(174, 93, 130, 0.18);
        color: #f7eef5;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeInPage 1s ease-in-out;
        min-height: 132px;
        height: 132px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        overflow: hidden;
        word-break: break-word;
    }
    .sub-card b {
        color: #fff7fc;
        font-size: 1rem;
        display: inline-block;
        margin-bottom: 0.35rem;
        line-height: 1.3;
    }
    .sub-card br {
        display: none;
    }
    .sub-card {
        line-height: 1.45;
        font-size: 0.94rem;
    }
    .sub-card span {
        display: block;
    }
    .sub-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 24px rgba(255, 122, 184, 0.16);
        border-color: #ffbfdc;
    }
    .section-note {
        font-size: 1rem;
        color: #f8eef6;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        background: rgba(35, 24, 33, 0.92);
        border: 1px solid #513547;
        border-radius: 18px;
        padding: 1rem 1.05rem;
    }
    .quote-box {
        background: rgba(35, 24, 33, 0.9);
        border-left: 4px solid #df5f97;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 0.55rem 0 1.15rem 0;
        color: #f8eef6;
        box-shadow: 0 10px 24px rgba(196, 95, 140, 0.1);
        animation: slideUp 0.8s ease-out;
    }
    .quote-box p {
        margin: 0.25rem 0;
        line-height: 1.5;
    }
    .section-title {
        font-size: 1.35rem;
        color: #fff7fc;
        font-weight: 700;
        margin: 0.7rem 0 0.25rem 0;
    }
    .section-subtitle {
        color: #ead8e4;
        margin-bottom: 0.9rem;
        line-height: 1.55;
    }
    .result-card {
        background: linear-gradient(180deg, rgba(35, 24, 33, 0.95) 0%, rgba(29, 20, 29, 0.95) 100%);
        border: 1px solid #513547;
        border-radius: 22px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 12px 26px rgba(193, 107, 146, 0.1);
        margin-top: 0.6rem;
        margin-bottom: 0.9rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }
    .mini-metric {
        background: rgba(29, 20, 29, 0.95);
        border: 1px solid #513547;
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
    }
    .mini-metric .label {
        font-size: 0.82rem;
        color: #e6d2df;
        margin-bottom: 0.2rem;
    }
    .mini-metric .value {
        font-size: 1.15rem;
        color: #fff7fc;
        font-weight: 700;
    }
    .floating-hearts {
        position: fixed;
        inset: 0;
        pointer-events: none;
        overflow: hidden;
        z-index: 0;
    }
    .floating-hearts span {
        position: absolute;
        bottom: -40px;
        font-size: 18px;
        color: rgba(255, 130, 180, 0.65);
        animation: floatHeart linear infinite;
    }
    .floating-hearts span:nth-child(1) { left: 8%; animation-duration: 13s; animation-delay: 0s; }
    .floating-hearts span:nth-child(2) { left: 20%; animation-duration: 16s; animation-delay: 2s; }
    .floating-hearts span:nth-child(3) { left: 34%; animation-duration: 12s; animation-delay: 1s; }
    .floating-hearts span:nth-child(4) { left: 49%; animation-duration: 15s; animation-delay: 4s; }
    .floating-hearts span:nth-child(5) { left: 63%; animation-duration: 14s; animation-delay: 3s; }
    .floating-hearts span:nth-child(6) { left: 77%; animation-duration: 17s; animation-delay: 5s; }
    .floating-hearts span:nth-child(7) { left: 89%; animation-duration: 13s; animation-delay: 1.5s; }
    /* Readability for Streamlit widgets */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stNumberInput label {
        color: #f8eef6 !important;
    }
    @keyframes fadeInPage {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatHeart {
        0% { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
        15% { opacity: 0.9; }
        50% { transform: translateY(-45vh) translateX(12px) scale(1); }
        100% { transform: translateY(-95vh) translateX(-10px) scale(1.1); opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="floating-hearts">
      <span>❤</span><span>❤</span><span>❤</span><span>❤</span><span>❤</span><span>❤</span><span>❤</span>
    </div>
    <div class="hero">
      <div class="eyebrow">Romantic Compatibility Experience</div>
      <h1>MatchCraft AI</h1>
      <p>Compare two people, reveal emotional alignment, and get elegant AI-powered insights that make match decisions feel clear, warm, and intentional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_a, top_b, top_c = st.columns(3, gap="medium")
with top_a:
    st.markdown(
        '<div class="sub-card"><b>Chemistry Score</b><span>A blended score that reflects shared vibe, behavior alignment, and match confidence.</span></div>',
        unsafe_allow_html=True,
    )
with top_b:
    st.markdown(
        '<div class="sub-card"><b>Love Insights</b><span>Natural-language guidance explaining why the connection feels strong or weak.</span></div>',
        unsafe_allow_html=True,
    )
with top_c:
    st.markdown(
        '<div class="sub-card"><b>Couple Comparison</b><span>A side-by-side view of traits, behavior, and compatibility differences.</span></div>',
        unsafe_allow_html=True,
    )

hero_media_left, hero_media_right = st.columns([1.15, 0.85])
with hero_media_left:
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <img src="https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?auto=format&fit=crop&w=1400&q=80"
             style="width:100%; max-height:320px; object-fit:cover; border-radius:18px; border:1px solid #513547; box-shadow:0 10px 26px rgba(174,93,130,0.18);" />
        """,
        unsafe_allow_html=True,
    )
with hero_media_right:
    st.markdown('<div class="section-title">Designed for modern dating</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">The interface is intentionally romantic and cinematic, with soft motion and elegant contrast so matching feels human, emotional, and memorable.</div>',
        unsafe_allow_html=True,
    )
st.markdown(
    """
    <div class="quote-box">
      <p><i>"Love is not just found. It is felt in the details."</i></p>
      <p><i>"When two hearts align, even small moments feel like magic."</i></p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-note">
      Fill both profiles and run the prediction to see compatibility, confidence, AI interpretation, and profile differences in one place.
      Strong matches usually show shared interests, similar social energy, and aligned communication behavior.
    </div>
    """,
    unsafe_allow_html=True,
)

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

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">The final decision combines pairwise compatibility with profile-level model confidence. Outcome: <b>{"Good Match" if pred == 1 else "Not Good Match"}</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-card">
          <div class="metric-grid">
            <div class="mini-metric">
              <div class="label">Final confidence</div>
              <div class="value">{confidence:.2%}</div>
            </div>
            <div class="mini-metric">
              <div class="label">Pairwise score</div>
              <div class="value">{pair_result['score']:.1f}/100</div>
            </div>
            <div class="mini-metric">
              <div class="label">User A profile score</div>
              <div class="value">{score_a:.2%}</div>
            </div>
            <div class="mini-metric">
              <div class="label">User B profile score</div>
              <div class="value">{score_b:.2%}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        f"Shared interests: **{', '.join(pair_result['shared_interests']) if pair_result['shared_interests'] else 'None'}**"
    )

    st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A concise explanation of relationship potential, strengths, and practical next-step suggestions.</div>',
        unsafe_allow_html=True,
    )
    st.info(get_ai_explanation(profile_a, profile_b, confidence, pred == 1))
    st.markdown('<div class="section-title">Comparison Table</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Use this side-by-side breakdown to spot where the two profiles naturally align and where they differ.</div>',
        unsafe_allow_html=True,
    )
    st.table(build_comparison_table(profile_a, profile_b))