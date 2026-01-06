"""
Streamlit frontend for Crop Disease Detection
Mobile-friendly interface with image upload, symptom checklist, and visualization
"""
import streamlit as st
import requests
import base64
import io
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Load translations
with open("src/translations.json", encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)

def t(key, lang):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# Page configuration
st.set_page_config(
    page_title="🌱 Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for light, clean, responsive layout
st.markdown("""
<style>
    :root {
        --bg: #f7f9fc;
        --card: #ffffff;
        --accent: #22c55e;
        --accent-2: #2563eb;
        --text: #0f172a;
        --muted: #475569;
        --border: rgba(15,23,42,0.08);
    }
    html, body, .stApp {
        background: radial-gradient(circle at 12% 20%, rgba(37,99,235,0.08), transparent 28%),
                    radial-gradient(circle at 82% 0%, rgba(34,197,94,0.12), transparent 30%),
                    var(--bg);
        color: var(--text);
        font-family: 'Inter', 'Manrope', 'Segoe UI', sans-serif;
    }
    .block-container {
        max-width: 1200px;
        padding: 1.5rem 1rem 3rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero */
    .hero {
        background: linear-gradient(120deg, rgba(37,99,235,0.10), rgba(34,197,94,0.14));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px 28px;
        box-shadow: 0 12px 34px rgba(15,23,42,0.12);
        margin-bottom: 18px;
    }
    .hero .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        color: var(--muted);
        margin: 0 0 6px;
    }
    .hero h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        color: #0b172a;
    }
    .hero .sub {
        margin: 8px 0 0;
        color: #1f2937;
        font-size: 15px;
    }

    /* Cards and panels */
    .card, [data-testid="stSidebar"] section, .stAlert { 
        background: var(--card) !important;
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        color: var(--text);
    }
    .stAlert { padding: 16px 18px; }
    .st-emotion-cache-16idsys p { color: var(--text); }

    /* Inputs */
    .stSelectbox, .stFileUploader, .stTextInput, .stTextArea { color: var(--text); }
    .stFileUploader > div { border: 1px dashed var(--border); border-radius: 12px; background: rgba(15,23,42,0.02); }
    .stTextInput > div > input, .stTextArea textarea {
        background: rgba(15,23,42,0.02);
        border-radius: 12px;
        border: 1px solid var(--border);
        color: var(--text);
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(120deg, var(--accent), #16a34a);
        color: #0b1221;
        font-weight: 700;
        border: none;
        padding: 12px 16px;
        box-shadow: 0 10px 24px rgba(22,163,74,0.25);
        transition: transform 0.08s ease, box-shadow 0.08s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(22,163,74,0.3);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* Metrics & expander tweaks */
    .stMetric { background: rgba(15,23,42,0.02); border-radius: 12px; padding: 12px; border: 1px solid var(--border); }
    .st-expander {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* Hide Streamlit heading anchor/link icons */
    .anchor-link {display: none !important;}
    /* Hide heading anchor/link icons only */
    [data-testid="stHeading"] a {display: none !important; pointer-events: none !important;}
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {display: none !important; pointer-events: none !important;}
    
    /* Prevent white screens and overlays - BALANCED APPROACH */
    .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
        filter: none !important; 
        opacity: 1 !important; 
        background: var(--bg) !important;
    }
    
    /* Hide spinners but keep content visible */
    [data-testid="stSpinner"] {
        opacity: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    /* Disable transitions only */
    * {
        transition: none !important;
        animation-duration: 0s !important;
    }
    
    /* Keep content always visible during reruns */
    [data-testid="stVerticalBlock"], 
    [data-testid="stHorizontalBlock"],
    [data-testid="stColumn"], 
    .element-container {
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Prevent form elements from causing flashes */
    div[data-testid="stTextInput"], 
    div[data-testid="stTextArea"],
    div[data-testid="stRadio"] {
        transition: none !important;
    }

    /* Mobile tweaks */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.6rem 2rem; }
        .hero { padding: 18px; }
        .hero h1 { font-size: 24px; }
        .hero .sub { font-size: 14px; }
        div.stButton > button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_SELECT_CROP_URL = f"{API_BASE_URL}/select_crop"
API_CROPS_URL = f"{API_BASE_URL}/crops"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(API_HEALTH_URL, timeout=5)
        return response.status_code == 200
    except:
        return False


def switch_crop_model(crop_name):
    """Switch to a different crop model"""
    try:
        response = requests.post(API_SELECT_CROP_URL, params={"crop": crop_name}, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error switching model: {e}")
        return None


def get_current_crop():
    """Get currently active crop model"""
    try:
        response = requests.get(API_CROPS_URL, timeout=5)
        if response.status_code == 200:
            return response.json().get("current", "potato")
        return "potato"
    except:
        return "potato"


def format_confidence(confidence):
    """Format confidence as percentage"""
    return f"{confidence:.1%}"


def get_severity_color(severity_level):
    """Get color for severity level"""
    severity_colors = {
        "Mild": "#90EE90",
        "Moderate": "#FFD700",
        "Severe": "#FF6B6B"
    }
    return severity_colors.get(severity_level, "#808080")


def get_severity_emoji(severity_level):
    """Get emoji for severity level"""
    severity_emoji = {
        "Mild": "🟢",
        "Moderate": "🟡",
        "Severe": "🔴"
    }
    return severity_emoji.get(severity_level, "⚪")


def translate_severity(severity_level: str, lang: str) -> str:
    if lang != "te":
        return severity_level
    mapping = {
        "Mild": t("severity_mild", lang),
        "Moderate": t("severity_moderate", lang),
        "Severe": t("severity_severe", lang),
    }
    return mapping.get(severity_level, severity_level)

DISEASE_TRANSLATIONS = {
    # Tomato
    "Tomato___Bacterial_spot": "టమోటా – బ్యాక్టీరియల్ స్పాట్",
    "Tomato___Early_blight": "టమోటా – ఎర్లీ బ్లైట్",
    "Tomato___Late_blight": "టమోటా – లేట్ బ్లైట్",
    "Tomato___Leaf_Mold": "టమోటా – లీఫ్ మోల్డ్",
    "Tomato___Septoria_leaf_spot": "టమోటా – సెప్టోరియా లీఫ్ స్పాట్",
    "Tomato___Target_Spot": "టమోటా – టార్గెట్ స్పాట్",
    "Tomato___healthy": "టమోటా – ఆరోగ్యంగా",
    # Potato
    "Potato___Early_blight": "బంగాళాదుంప – ఎర్లీ బ్లైట్",
    "Potato___Late_blight": "బంగాళాదుంప – లేట్ బ్లైట్",
    "Potato___healthy": "బంగాళాదుంప – ఆరోగ్యంగా",
    # Grape
    "Grape___Black_rot": "ద్రాక్ష – బ్లాక్ రాట్",
    "Grape___Esca_(Black_Measles)": "ద్రాక్ష – ఎస్కా (బ్లాక్ మీజిల్స్)",
    "Grape___healthy": "ద్రాక్ష – ఆరోగ్యంగా",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "ద్రాక్ష – లీఫ్ బ్లైట్",
    # Apple
    "Apple___Apple_scab": "ఆపిల్ – ఆపిల్ స్కాబ్",
    "Apple___Black_rot": "ఆపిల్ – బ్లాక్ రాట్",
    "Apple___Cedar_apple_rust": "ఆపిల్ – సీడర్ ఆపిల్ రస్ట్",
    "Apple___healthy": "ఆపిల్ – ఆరోగ్యంగా",
}

# Common treatment text translations
TREATMENT_TRANSLATIONS = {
    "Remove infected leaves": "సంక్రమిత ఆకులను తొలగించండి",
    "Copper fungicide": "కాపర్ ఫంగిసైడ్",
    "Mulch": "మల్చ్ వాడండి",
    "Chlorothalonil": "క్లోరోథలోనిల్",
    "Mancozeb": "మాంకోజెబ్",
    "Crop rotation": "పంట మార్పిడి",
    "Good air circulation": "ఉత్తమ గాలి ప్రసరణ",
    "Remove debris": "చెత్తను తొలగించండి",
}


def format_disease_name(raw_name: str, lang: str = "en") -> str:
    """Human-friendly disease label from raw class name, with optional Telugu."""
    if not raw_name:
        return "Unknown"
    pretty = ""
    if "___" in raw_name:
        crop, disease = raw_name.split("___", 1)
    else:
        crop, disease = "", raw_name
    crop = crop.replace("_", " ").strip().title()
    disease = disease.replace("_", " ").strip().title()
    pretty = f"{crop} – {disease}" if crop else disease
    if lang == "te":
        # Prefer raw-name lookup; fall back to pretty string if provided
        if raw_name in DISEASE_TRANSLATIONS:
            return DISEASE_TRANSLATIONS[raw_name]
        if pretty in DISEASE_TRANSLATIONS:
            return DISEASE_TRANSLATIONS[pretty]
    return pretty


def display_disease_info(prediction_result, lang):
    """Display disease detection result simply"""
    col1, col2 = st.columns([3, 2])
    
    with col1:
        pretty_name = format_disease_name(prediction_result['disease'], lang)
        st.markdown(f"## 🦠 {pretty_name}")
    
    with col2:
        severity_level = prediction_result['severity_level']
        severity_emoji = get_severity_emoji(severity_level)
        st.markdown(f"### {severity_emoji} {translate_severity(severity_level, lang)}")


def display_confidence_bar(confidence, lang, threshold=0.70):
    """Display confidence as progress bar"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Normalize for color coding
        if confidence < threshold:
            color = "#FF6B6B"  # Red
            status = t("status_low", lang)
        elif confidence < 0.85:
            color = "#FFD700"  # Yellow
            status = t("status_medium", lang)
        else:
            color = "#90EE90"  # Green
            status = t("status_high", lang)
        
        st.markdown(f"<div style='background-color: #f0f2f6; border-radius: 10px; padding: 10px;'>"
                   f"<div style='background-color: {color}; width: {confidence*100}%; padding: 10px; border-radius: 5px;'>"
                   f"<b>{format_confidence(confidence)}</b></div></div>",
                   unsafe_allow_html=True)
    
    with col2:
        st.write(status)


def display_treatment_advice(treatment_advice, lang):
    """Display treatment recommendations in structured format"""
    st.markdown(f"### {t('treatment_recommendations', lang)}")
    
    # Cause
    st.markdown(f"**{t('cause', lang)}:**")
    st.write(treatment_advice.get('cause', 'Unknown'))
    
    # Organic treatment
    with st.expander(t('organic_treatment', lang), expanded=False):
        for treatment in treatment_advice.get('organic_treatment', []):
            if lang == "te":
                treatment = TREATMENT_TRANSLATIONS.get(treatment, treatment)
            st.write(f"• {treatment}")
    
    # Chemical treatment
    with st.expander(t('chemical_treatment', lang), expanded=False):
        for treatment in treatment_advice.get('chemical_treatment', []):
            if lang == "te":
                treatment = TREATMENT_TRANSLATIONS.get(treatment, treatment)
            st.write(f"• {treatment}")
    
    # Prevention
    with st.expander(t('prevention_tips', lang), expanded=True):
        for tip in treatment_advice.get('prevention', []):
            if lang == "te":
                tip = TREATMENT_TRANSLATIONS.get(tip, tip)
            st.write(f"• {tip}")
    
    # Expert consultation warning
    if 'expert_consultation' in treatment_advice:
        st.warning(treatment_advice['expert_consultation'])


def display_symptom_analysis(symptom_analysis, lang):
    """Display symptom fusion results"""
    if not symptom_analysis:
        return
    
    st.markdown(f"### {t('symptom_analysis', lang)}")
    
    active = symptom_analysis.get('active_symptoms', [])
    supporting = symptom_analysis.get('supporting_symptoms', [])
    contradicting = symptom_analysis.get('contradicting_symptoms', [])
    adjustment = symptom_analysis.get('confidence_adjustment', 0.0)
    reason = symptom_analysis.get('reason', '')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**{t('active_symptoms', lang)}:** {len(active)}")
        if active:
            for symptom in active:
                st.write(f"✓ {symptom.replace('_', ' ').title()}")
    
    with col2:
        st.write(f"**{t('supporting_symptoms', lang)}:** {len(supporting)}")
        if supporting:
            for symptom in supporting:
                st.write(f"✅ {symptom.replace('_', ' ').title()}")
    
    with col3:
        st.write(f"**{t('contradicting_symptoms', lang)}:** {len(contradicting)}")
        if contradicting:
            for symptom in contradicting:
                st.write(f"❌ {symptom.replace('_', ' ').title()}")
    
    if adjustment != 0.0:
        st.info(f"**{t('confidence_adjustment', lang)}:** {adjustment:+.1%}\n\n{reason}")


def display_grad_cam(heatmap_base64, original_image):
    """Display Grad-CAM heatmap visualization"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image:**")
        st.image(original_image, width=280)
    
    with col2:
        st.markdown("**Grad-CAM Heatmap:**")
        if heatmap_base64:
            heatmap_image = Image.open(io.BytesIO(base64.b64decode(heatmap_base64)))
            st.image(heatmap_image, width=280)
            st.caption("🔴 Red regions = high influence on prediction")


def display_image_quality(quality):
    """Show blur/lighting diagnostics."""
    if not quality:
        return
    issues = quality.get("issues", [])
    blur = quality.get("blur_score")
    bright = quality.get("brightness")
    cols = st.columns(3)
    cols[0].metric("Blur score", f"{blur:.0f}")
    cols[1].metric("Brightness", f"{bright:.0f}")
    cols[2].write("Issues" if issues else "Looks good")
    if issues:
        st.warning(" | ".join(issues))
    else:
        st.success("Image quality looks good")


def photo_tips_card(lang):
    """Quick capture tips."""
    with st.expander(t("photo_tips", lang), expanded=False):
        st.write(t("photo_tip1", lang))
        st.write(t("photo_tip2", lang))
        st.write(t("photo_tip3", lang))
        st.write(t("photo_tip4", lang))


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    # Initialize session state for results persistence
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state['uploaded_file_name'] = None

    # Language selection (default English)
    lang = st.sidebar.selectbox(
        "🌐 Language / భాష",
        options=[("en", "English"), ("te", "తెలుగు")],
        format_func=lambda x: x[1],
        key="lang_selector"
    )[0]

    # Hero header
    st.markdown(
        f"""
        <div class="hero">
            <p class="eyebrow">Smart Agri Health</p>
            <h1>{t('title', lang)}</h1>
            <p class="sub">{t('subtitle', lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check API health
    if not check_api_health():
        st.error(t("api_not_running", lang))
        st.stop()

    # Get current crop from backend
    current_crop = get_current_crop()
    
    # Initialize session state for crop
    if 'active_crop' not in st.session_state:
        st.session_state.active_crop = current_crop

    # Crop selection (top of page)
    st.markdown(f"### {t('select_crop', lang)}")
    col1, col2 = st.columns([2, 3])

    with col1:
        selected_crop = st.selectbox(
            t("choose_crop", lang),
            options=["potato", "grape", "apple"],
            format_func=lambda x: {"potato": "🥔 Potato", "grape": "🍇 Grape", "apple": "🍎 Apple"}[x],
            index=["potato", "grape", "apple"].index(current_crop) if current_crop in ["potato", "grape", "apple"] else 0,
            key="crop_selector"
        )
    
    with col2:
        # Auto-switch model when selection changes
        if selected_crop != st.session_state.active_crop:
            with st.spinner(f"Switching to {selected_crop} model..."):
                result = switch_crop_model(selected_crop)
                if result and result.get("status") == "success":
                    st.session_state.active_crop = selected_crop
                    st.success(f"✓ {selected_crop.title()} model loaded ({result.get('classes')} classes, {result.get('accuracy')} accuracy)")
                    st.rerun()
                else:
                    st.error(f"Failed to switch to {selected_crop} model")
        else:
            # Show current model info
            crop_emoji = {"potato": "🥔", "grape": "🍇", "apple": "🍎"}[st.session_state.active_crop]
            accuracy_map = {"potato": "95.43%", "grape": "97.53%", "apple": "94.32%"}
            accuracy = accuracy_map.get(st.session_state.active_crop, "N/A")
            st.info(f"{crop_emoji} **Active Model:** {st.session_state.active_crop.title()} ({accuracy} accuracy)")

    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.write("**Model:** MobileNetV2 + Transfer Learning")
        st.write("**Input Size:** 224×224 RGB")
        st.write("**Explainability:** Grad-CAM")

        st.markdown("---")
        st.markdown(t("how_to_use", lang))
        st.write("\n".join([
            t("step1", lang),
            t("step2", lang),
            t("step3", lang),
            t("step4", lang),
            t("step5", lang),
        ]))
        photo_tips_card(lang)

    # Main content - Two column layout
    col_upload, col_symptoms = st.columns([2, 1])

    # Image upload column
    with col_upload:
        st.markdown(f"### {t('upload_leaf', lang)}")
        uploaded_file = st.file_uploader(
            t('choose_image', lang),
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Mobile camera photos work best",
            key="leaf_uploader"
        )

        # Check if a new image was uploaded (clear previous results)
        if uploaded_file and uploaded_file.name != st.session_state.get('uploaded_file_name'):
            st.session_state['prediction_result'] = None
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['feedback_submitted'] = False

        analyze_clicked = False
        status_placeholder = st.empty()
        if uploaded_file:
            # Display uploaded image (reduced size)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            # Analyze button sized to this column
            analyze_clicked = st.button(t('analyze_image', lang), use_container_width=True)

    # Symptoms column
    with col_symptoms:
        st.markdown(f"### {t('visible_symptoms', lang)}")
        st.write(t('check_symptoms', lang))

        yellowing_leaves = st.checkbox(t('yellowing_leaves', lang))
        brown_spots = st.checkbox(t('brown_spots', lang))
        wilting = st.checkbox(t('wilting', lang))
        white_fungal = st.checkbox(t('white_fungal', lang))

        symptoms = {
            "yellowing_leaves": yellowing_leaves,
            "brown_spots": brown_spots,
            "wilting": wilting,
            "white_fungal_growth": white_fungal
        }

        symptom_count = sum(symptoms.values())
        if symptom_count > 0:
            st.info(f"📌 {symptom_count} symptom(s) selected")

    # Prediction button and results
    if uploaded_file and analyze_clicked:
        # Create a placeholder for status within upload column width
        status_placeholder.info(t('analyzing', lang))

        try:
            # Prepare request
            files = {'image': uploaded_file.getvalue()}
            data = {
                'yellowing_leaves': symptoms['yellowing_leaves'],
                'brown_spots': symptoms['brown_spots'],
                'wilting': symptoms['wilting'],
                'white_fungal_growth': symptoms['white_fungal_growth']
            }

            # Send request
            # Increase timeout because explainability and larger models take longer
            response = requests.post(API_PREDICT_URL, files=files, data=data, timeout=90)
            status_placeholder.empty()  # Clear status message

            if response.status_code == 200:
                prediction_result = response.json()
                # Store in session state for persistence across reruns
                st.session_state['prediction_result'] = prediction_result

                # Display results
                st.markdown("---")
                st.markdown(f"## {t('prediction_results', lang)}")

                # Disease and severity
                display_disease_info(prediction_result, lang)

                st.markdown("---")

                # Confidence and severity in one row
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {t('confidence_level', lang)}")
                    display_confidence_bar(
                        prediction_result['confidence'],
                        lang,
                        threshold=0.70
                    )

                with col2:
                    st.markdown(f"### 📊 {t('severity_score', lang)}")
                    st.metric(
                        "",
                        f"{prediction_result['severity_score']:.0f}/100"
                    )

                st.markdown(f"### {t('image_quality', lang)}")
                display_image_quality(prediction_result.get("image_quality"))

                st.markdown("---")

                # Treatment advice
                display_treatment_advice(prediction_result['treatment_advice'], lang)

                # Symptom analysis (only if user selected symptoms)
                if symptom_count > 0 and prediction_result.get('symptom_analysis'):
                    st.markdown(f"### {t('symptom_analysis', lang)}")
                    display_symptom_analysis(prediction_result['symptom_analysis'], lang)
                    st.markdown("---")

                # Grad-CAM visualization
                st.markdown(f"## {t('explainable_ai', lang)}")
                st.write(prediction_result.get('explanation', t('explanation_help', lang)))

                if prediction_result.get('heatmap_base64'):
                    display_grad_cam(
                        prediction_result['heatmap_base64'],
                        uploaded_file
                    )
                else:
                    st.warning(t('heatmap_warning', lang))

                st.markdown("---")

                # Model info
                with st.expander(t('model_information', lang)):
                    for key, value in prediction_result['model_info'].items():
                        st.write(f"**{key}:** {value}")

                with st.expander(t('was_helpful', lang)):
                    helpful_options = ["Yes", "No", "Unsure"]
                    helpful = st.radio(
                        t('prediction_usefulness', lang),
                        helpful_options,
                        format_func=lambda x: t(x.lower(), lang),
                        key="feedback_helpful",
                        horizontal=True
                    )
                    user_label = st.text_input(t('user_label_prompt', lang), key="feedback_label")
                    comments = st.text_area(t('comments_prompt', lang), height=80, key="feedback_comments")
                    if st.button(t('submit_feedback', lang)):
                        try:
                            feedback_payload = {
                                "prediction": prediction_result['disease'],
                                "user_label": st.session_state.get('feedback_label') or None,
                                "helpful": True if st.session_state.get('feedback_helpful') == "Yes" else False if st.session_state.get('feedback_helpful') == "No" else None,
                                "comments": st.session_state.get('feedback_comments') or None,
                                "confidence": prediction_result['confidence']
                            }
                            res = requests.post(f"{API_BASE_URL}/feedback", json=feedback_payload, timeout=5)
                            if res.status_code == 200:
                                st.session_state['feedback_submitted'] = True
                                st.success(t('feedback_received', lang))
                            else:
                                st.warning(t('feedback_warning', lang))
                        except Exception as exc:
                            st.warning(f"{t('feedback_error', lang)}{exc}")

            else:
                st.error(f"{t('prediction_failed', lang)}: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.Timeout:
            status_placeholder.empty()
            st.error(t('request_timeout', lang))
        except requests.exceptions.ConnectionError:
            status_placeholder.empty()
            st.error(t('connection_error', lang))
        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ Error: {str(e)}")

    # Display stored results if available (for feedback form persistence)
    elif uploaded_file and st.session_state.get('prediction_result'):
        prediction_result = st.session_state['prediction_result']
        symptom_count = sum([yellowing_leaves, brown_spots, wilting, white_fungal])
        
        # Display results
        st.markdown("---")
        st.markdown(f"## {t('prediction_results', lang)}")

        # Disease and severity
        display_disease_info(prediction_result, lang)

        st.markdown("---")

        # Confidence and severity in one row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {t('confidence_level', lang)}")
            display_confidence_bar(
                prediction_result['confidence'],
                lang,
                threshold=0.70
            )

        with col2:
            st.markdown(f"### 📊 {t('severity_score', lang)}")
            st.metric(
                "",
                f"{prediction_result['severity_score']:.0f}/100"
            )

        st.markdown(f"### {t('image_quality', lang)}")
        display_image_quality(prediction_result.get("image_quality"))

        st.markdown("---")

        # Treatment advice
        display_treatment_advice(prediction_result['treatment_advice'], lang)

        # Symptom analysis (only if user selected symptoms)
        if symptom_count > 0 and prediction_result.get('symptom_analysis'):
            st.markdown(f"### {t('symptom_analysis', lang)}")
            display_symptom_analysis(prediction_result['symptom_analysis'], lang)
            st.markdown("---")

        # Grad-CAM visualization
        st.markdown(f"## {t('explainable_ai', lang)}")
        st.write(prediction_result.get('explanation', t('explanation_help', lang)))

        if prediction_result.get('heatmap_base64'):
            display_grad_cam(
                prediction_result['heatmap_base64'],
                uploaded_file
            )
        else:
            st.warning(t('heatmap_warning', lang))

        st.markdown("---")

        # Model info
        with st.expander(t('model_information', lang)):
            for key, value in prediction_result['model_info'].items():
                st.write(f"**{key}:** {value}")

        # Show feedback form only if not submitted
        if st.session_state.get('feedback_submitted'):
            st.success(t('feedback_received', lang))
        else:
            with st.expander(t('was_helpful', lang), expanded=False):
                # Use session state keys to prevent form structure changes
                helpful_options = ["Yes", "No", "Unsure"]
                helpful = st.radio(
                    t('prediction_usefulness', lang),
                    helpful_options,
                    format_func=lambda x: t(x.lower(), lang),
                    key="fb_helpful",
                    horizontal=True
                )
                user_label = st.text_input(
                    t('user_label_prompt', lang),
                    key="fb_label"
                )
                comments = st.text_area(
                    t('comments_prompt', lang),
                    height=80,
                    key="fb_comments"
                )
                
                submit_clicked = st.button(t('submit_feedback', lang), type="primary", use_container_width=True, key="fb_submit")
                
                if submit_clicked:
                    try:
                        feedback_payload = {
                            "prediction": prediction_result['disease'],
                            "user_label": st.session_state.get('fb_label') or None,
                            "helpful": True if st.session_state.get('fb_helpful') == "Yes" else False if st.session_state.get('fb_helpful') == "No" else None,
                            "comments": st.session_state.get('fb_comments') or None,
                            "confidence": prediction_result['confidence']
                        }
                        res = requests.post(f"{API_BASE_URL}/feedback", json=feedback_payload, timeout=10)
                        if res.status_code == 200:
                            st.session_state['feedback_submitted'] = True
                            st.success(t('feedback_received', lang))
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(t('feedback_warning', lang))
                    except Exception as exc:
                        st.error(f"{t('feedback_error', lang)}{exc}")

    # Empty prompt if no image
    elif not uploaded_file:
        st.info(t('upload_to_start', lang))


if __name__ == "__main__":
    main()
