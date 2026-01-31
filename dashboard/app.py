import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/analyze_video"

st.set_page_config(page_title="YouTube Sentiment Insights", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF0000;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #CC0000;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .insight-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎥 YouTube Emotion & Insight Analysis")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    analyze_button = st.button("Analyze Video Sentiment")

if analyze_button:
    if not url_input:
        st.warning("Please enter a valid YouTube URL first.")
    else:
        with st.spinner("Analyzing comments... This may take a few seconds."):
            try:
                payload = {
                    "url": url_input,
                    "max_comments": 20,
                    "threshold": 0.05
                }
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(f"❌ {data['error']}")
                    else:
                        st.success("Analysis Complete!")
                        
                        st.subheader("📝 Video Review & Insights")
                        if "video_review" in data:
                            st.info(data["video_review"])
                        else:
                            st.info("No detailed review generated.")
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Comments Analyzed", data.get("total_comments_analyzed", 0))
                        
                        st.subheader("📊 Dominant Emotions")
                        dominant = data.get("dominant_emotions", [])
                        if dominant:
                            st.write(", ".join([f"**{e}**" for e in dominant]))
                        else:
                            st.write("No strong emotions detected.")
                        
                        st.subheader("💬 Top Comments Boosting Dominant Emotions")
                        supporting = data.get("top_supporting_comments", [])
                        if supporting:
                            for i, comment in enumerate(supporting, 1):
                                st.markdown(f"**{i}.** {comment}")
                        else:
                            st.write("No specific supporting comments identified.")
                                
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: Could not reach the API. Make sure the FastAPI server is running. ({e})")

with col2:
    st.markdown("""
    ### How it works
    1. Paste a YouTube link.
    2. We fetch the top **relevant comments**.
    3. Our **Sentiment AI** (RoBERTa) predicts emotions like Admiration, Joy, or Disappointment.
    4. We generate a **Real-World Insight** report for you.
    ---
    ### Note on Accuracy
    We have adjusted our AI to reduce **Neutral Bias** and focus on meaningful emotional drivers. 
    """)
    
    if 'data' in locals() and "error" not in data:
         st.write("### Dominant High-Level Emotions")
         for emotion in data.get("dominant_emotions", []):
             st.write(f"- {emotion}")