import streamlit as st
import google.generativeai as genai
import pdfplumber
import json
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="FinSight AI ⚡ Gemini Edition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- App Styling (Optional but Recommended) ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    .stAlert {
        border-radius: 0.5rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini API Configuration ---
# For Streamlit Community Cloud, secrets are stored in st.secrets.
# The key should be named "google_api_key".
try:
    genai.configure(api_key=st.secrets["google_api_key"])
except (KeyError, FileNotFoundError):
    st.error("⚠️ **Warning:** Google API key not found in Streamlit secrets.", icon="🚨")
    st.info("Please add your Google API key to your Streamlit secrets to continue. Name it `google_api_key`.")
    st.stop()


# --- Core Functions ---

@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(uploaded_file):
    """Extracts all text from an uploaded PDF file."""
    if uploaded_file is None:
        return ""
    
    # Use BytesIO to handle the uploaded file in memory
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    full_text = ""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

@st.cache_data(show_spinner="Gemini is analyzing the document...")
def analyze_financial_text_with_gemini(_text_chunk):
    """
    Uses Google Gemini 1.5 Flash to analyze financial text.
    The _text_chunk is used for caching; the full text is passed in the prompt.
    """
    # The model name for Gemini 1.5 Flash
    model = genai.GenerativeModel('gemini-2.0-flash')

    # This prompt is engineered to force a JSON output.
    prompt = f"""
    You are an expert financial analyst AI named FinSight, powered by Google Gemini.
    Your task is to analyze the financial text from a company's report.
    
    **Instructions**:
    1.  **Executive Summary**: Write a concise, professional summary of the company's financial health, performance, and position based on the provided text. Highlight key strengths, weaknesses, opportunities, and threats if they are apparent.
    2.  **Key Ratios**: Calculate the following financial ratios if the data is available. If a value cannot be calculated from the text, return "N/A".
        - Current Ratio (Current Assets / Current Liabilities)
        - Debt-to-Equity Ratio (Total Liabilities / Total Equity)
        - Net Profit Margin (Net Income / Total Revenue)
        - Return on Equity (ROE) (Net Income / Total Equity)
    3.  **Data Extraction**: Extract the key figures for the most recent year available from the Income Statement and Balance Sheet. If a specific figure is not found, its value should be null.

    Your entire response **MUST** be a single, valid JSON object. Do not include any text or markdown before or after the JSON object.

    The required JSON structure is:
    {{
      "executive_summary": "Your detailed summary here.",
      "key_ratios": {{
        "Current Ratio": "X.XX",
        "Debt-to-Equity Ratio": "X.XX",
        "Net Profit Margin": "XX.X%",
        "Return on Equity (ROE)": "XX.X%"
      }},
      "extracted_data": {{
        "Income Statement": {{
          "Total Revenue": 1000000,
          "Net Income": 100000
        }},
        "Balance Sheet": {{
          "Total Current Assets": 500000,
          "Total Current Liabilities": 250000,
          "Total Liabilities": 400000,
          "Total Stockholders’ Equity": 600000
        }}
      }}
    }}
    
    **Financial Text to Analyze**:
    ---
    {_text_chunk} 
    ---
    
    Now, provide the JSON response.
    """
    try:
        # Gemini's feature to force JSON output
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        # The response.text will be a valid JSON string
        result_json = json.loads(response.text)
        return result_json
    except Exception as e:
        st.error(f"Error during AI analysis with Gemini: {e}", icon="🔥")
        return None

# --- Main App Interface ---

st.title("FinSight AI ⚡ Gemini Edition")
st.markdown("Upload a company's financial report (PDF) to receive an instant, AI-powered analysis.")

# Using session state to store analysis results and file name
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None

uploaded_file = st.file_uploader(
    "Choose a financial document...",
    type="pdf",
    help="Upload an annual report or financial statement in PDF format."
)

if uploaded_file is not None:
    # Clear previous results if a new file is uploaded
    if st.session_state.last_file_name != uploaded_file.name:
        st.session_state.analysis_result = None
        st.session_state.last_file_name = uploaded_file.name

    if st.button("Analyze Financials", type="primary", use_container_width=True):
        full_text = extract_text_from_pdf(uploaded_file)
        if full_text and len(full_text) > 100: # Basic check for valid text
            # The full_text is passed to the function, but caching is based on its content
            analysis_result = analyze_financial_text_with_gemini(full_text)
            st.session_state.analysis_result = analysis_result
        else:
            st.error("Could not extract sufficient text from the document. Please try another file.", icon="📄")
            st.session_state.analysis_result = None

# --- Display Results ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    st.markdown("---")
    st.header("Financial Analysis Report")

    # 1. Executive Summary
    st.subheader("📝 Executive Summary")
    st.info(result.get("executive_summary", "Summary could not be generated."))

    # 2. Key Ratios in columns
    st.subheader("📊 Key Financial Ratios")
    key_ratios = result.get("key_ratios", {})
    if key_ratios:
        cols = st.columns(len(key_ratios))
        for i, (key, value) in enumerate(key_ratios.items()):
            with cols[i]:
                st.metric(label=key, value=str(value))
    else:
        st.warning("No key ratios could be calculated from the document.")

    # 3. Extracted Data in an expander
    with st.expander("🔬 View Raw Extracted Data (JSON)"):
        st.json(result.get("extracted_data", {"error": "No data extracted."}))

st.markdown("---")
st.markdown("Powered by **Google Gemini 1.5 Flash** and **Streamlit**.")
