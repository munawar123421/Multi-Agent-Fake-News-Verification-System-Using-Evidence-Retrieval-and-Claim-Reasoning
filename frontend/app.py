"""Streamlit frontend for fake news verification system."""
import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fake News Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .verdict-support {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #155724;
    }
    .verdict-refute {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #721c24;
    }
    .verdict-not-enough {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #856404;
    }
    .evidence-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.title("🔍 Multi-Agent Fake News Verification System")
st.markdown("**Production-Quality Fact-Checking with LangGraph**")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses a **multi-agent architecture** to verify factual claims:
    
    1. **Claim Agent** - Extracts and normalizes claims
    2. **Query Agent** - Generates search queries
    3. **Retrieval Agent** - Retrieves evidence
    4. **Ranking Agent** - Ranks by relevance
    5. **Reasoning Agent** - Performs verification
    6. **Explanation Agent** - Generates explanations
    """)
    
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API is healthy")
            else:
                st.error("❌ API returned error")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")

# Main content
tab1, tab2 = st.tabs(["Verify Claim", "System Info"])

with tab1:
    # Claim input section
    st.header("Enter Claim")
    claim = st.text_area(
        "Enter the claim you want to verify:",
        placeholder="e.g., COVID vaccines contain microchips",
        height=100
    )
    
    verify_button = st.button("🔍 Verify Claim", type="primary", use_container_width=True)
    
    if verify_button and claim:
        with st.spinner("Verifying claim... This may take a moment."):
            try:
                # Make API request
                response = requests.post(
                    f"{api_url}/verify",
                    json={"claim": claim},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Results Dashboard
                    st.success("✅ Verification Complete!")
                    
                    st.markdown("---")
                    
                    # Verdict Badge
                    verdict = result["verdict"]
                    confidence = result["confidence"]
                    
                    if verdict == "SUPPORT":
                        verdict_class = "verdict-support"
                        verdict_emoji = "✅"
                    elif verdict == "REFUTE":
                        verdict_class = "verdict-refute"
                        verdict_emoji = "❌"
                    else:
                        verdict_class = "verdict-not-enough"
                        verdict_emoji = "⚠️"
                    
                    st.markdown(
                        f'<div class="{verdict_class}">{verdict_emoji} {verdict}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence meter
                    st.subheader("Confidence Score")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "lightblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Explanation Panel
                    st.subheader("📝 Explanation")
                    st.info(result["explanation"])
                    
                    st.markdown("---")
                    
                    # Evidence Panel
                    st.subheader("📚 Evidence")
                    evidence = result.get("evidence", [])
                    
                    if evidence:
                        for i, ev in enumerate(evidence, 1):
                            with st.expander(f"Evidence {i}: {ev['title'][:60]}... (Relevance: {ev.get('relevance_score', 0):.2f})"):
                                st.markdown(f"**Source:** {ev['source']}")
                                if ev.get('url'):
                                    st.markdown(f"**URL:** [{ev['url']}]({ev['url']})")
                                st.markdown(f"**Relevance Score:** {ev.get('relevance_score', 0):.3f}")
                                st.markdown("**Content:**")
                                content = ev.get('snippet') or ev.get('text', '')
                                st.text(content[:500] + "..." if len(content) > 500 else content)
                    else:
                        st.warning("No evidence found")
                    
                    st.markdown("---")
                    
                    # Reasoning Panel
                    st.subheader("🧠 Reasoning")
                    with st.expander("View Detailed Reasoning", expanded=False):
                        st.markdown(result.get("reasoning", "No reasoning available"))
                    
                    st.markdown("---")
                    
                    # Agent Trace Panel
                    st.subheader("🤖 Agent Execution Trace")
                    trace = result.get("execution_trace", [])
                    
                    if trace:
                        for step in trace:
                            status_emoji = "✅" if step["status"] == "completed" else "❌" if step["status"] == "failed" else "⏳"
                            duration = f" ({step.get('duration_ms', 0):.0f}ms)" if step.get('duration_ms') else ""
                            st.text(f"{status_emoji} {step['agent']} - {step['status']}{duration}")
                            
                            if step.get('details'):
                                with st.expander(f"Details for {step['agent']}"):
                                    st.json(step['details'])
                    
                    # Sources
                    if result.get("sources"):
                        st.markdown("---")
                        st.subheader("🔗 Sources")
                        for source in result["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']}) - {source.get('source', 'Unknown')}")
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. The verification is taking too long.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif verify_button:
        st.warning("Please enter a claim to verify")

with tab2:
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **Multi-Agent Workflow:**
        - ClaimAgent → QueryAgent
        - QueryAgent → RetrievalAgent
        - RetrievalAgent → RankingAgent
        - RankingAgent → ReasoningAgent
        - ReasoningAgent → ExplanationAgent
        
        **Technology Stack:**
        - Backend: FastAPI
        - Agents: LangGraph
        - LLM: Gemini/OpenAI
        - Embeddings: sentence-transformers
        - Vector Search: FAISS
        - Frontend: Streamlit
        """)
    
    with col2:
        st.subheader("Workflow Diagram")
        st.markdown("""
        ```
        User Claim
           ↓
        Claim Agent
           ↓
        Query Agent
           ↓
        Retrieval Agent
           ↓
        Ranking Agent
           ↓
        Reasoning Agent
           ↓
        Explanation Agent
           ↓
        Results
        ```
        """)
    
    st.markdown("---")
    st.subheader("API Endpoints")
    st.code("""
    POST /verify - Verify a claim
    GET /health - Health check
    GET /metrics - System metrics
    GET /graph - Workflow visualization
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Multi-Agent Fake News Verification System v1.0.0"
    "</div>",
    unsafe_allow_html=True
)
