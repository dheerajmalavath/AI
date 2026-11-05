import streamlit as st
import os
import yaml
from src.ai_model import get_nutrition_info, query_nutrition_knowledge
from src.db import save_query, get_history, clear_history

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriSense",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Minimal table + history styling ─────────────────────────────────────────
st.markdown("""
<style>
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
thead th {
    background-color: #1a7a32; color: #fff;
    padding: 0.6rem 1rem; text-align: left; font-weight: 600;
}
tbody td { padding: 0.5rem 1rem; border-bottom: 1px solid #e4e9e4; }
tbody tr:nth-child(even) td { background-color: #f4f7f4; }
tbody tr:last-child td { border-bottom: none; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load config ───────────────────────────────────────────────────────────────
config = {}
if os.path.exists("config/config.yaml"):
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        pass

model_name  = config.get("api", {}).get("model", "models/gemini-1.5-flash")
temperature = config.get("api", {}).get("temperature", 0.7)
api_key     = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("NutriSense")
st.sidebar.caption("AI-powered nutrition analyzer")
st.sidebar.divider()

st.sidebar.markdown("**Model**")
st.sidebar.text(model_name)
st.sidebar.markdown("**Temperature**")
st.sidebar.text(str(temperature))
st.sidebar.divider()

st.sidebar.markdown("**API Status**")
if api_key and "your_" not in api_key.lower():
    st.sidebar.success("Connected")
else:
    st.sidebar.error("API key missing — check .env")

st.sidebar.divider()
st.sidebar.markdown("**Knowledge Base**")
st.sidebar.text("data/  (local files)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Food Nutrition Analyzer")
st.write("Search nutrients for any food item, or ask a question from the knowledge base.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Nutrition Lookup", "Knowledge Base", "Query History"])

# ── Tab 1: Nutrition Lookup ───────────────────────────────────────────────────
with tab1:
    st.subheader("Nutrition Lookup")
    st.write("Enter a food item to get a full breakdown — calories, macros, micros, and health notes.")

    food_item = st.text_input(
        "Food item",
        placeholder="e.g. Apple, Grilled Salmon, Avocado Toast",
        key="food_input"
    )

    if st.button("Analyze", key="btn_analyze", type="primary"):
        if food_item.strip():
            with st.spinner(f"Analyzing {food_item}..."):
                result = get_nutrition_info(food_item)
            save_query("nutrition", food_item, result)
            st.divider()
            st.markdown(result)
        else:
            st.warning("Please type a food item above.")

# ── Tab 2: Knowledge Base ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Knowledge Base Q&A")
    st.write("Ask any nutrition question — answers are pulled from the local documents in `data/`.")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the recommended daily calorie intake?",
        key="kb_input"
    )

    if st.button("Search", key="btn_query", type="primary"):
        if question.strip():
            with st.spinner("Searching..."):
                result = query_nutrition_knowledge(question)
            save_query("knowledge", question, result)
            st.divider()
            st.markdown(result)
        else:
            st.warning("Please type a question above.")

# ── Tab 3: Query History ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Query History")
    st.write("All past searches and answers saved to the local SQLite database.")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear History", key="btn_clear"):
            clear_history()
            st.success("History cleared.")
            st.rerun()

    history = get_history(limit=50)

    if not history:
        st.info("No history yet — run a nutrition lookup or knowledge base search first.")
    else:
        for row in history:
            badge = "🍎 Nutrition" if row["query_type"] == "nutrition" else "📖 Knowledge"
            with st.expander(f"{badge}  —  **{row['query']}**  `{row['created_at']}`"):
                st.markdown(row["response"])
