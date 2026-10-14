import base64
import hashlib
import io
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras


# =========================================================
# PATHS & CONSTANTS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "traffic_sign_model.keras"

IMG_SIZE = 32
TEST_ACCURACY = "95.59%"


# =========================================================
# ICONS
# - SVG inline (style Lucide) pour le HTML personnalisé
# - :material/nom: pour les widgets natifs Streamlit
# =========================================================

ICON_PATHS = {
    "signpost": '<path d="M12 3v3"/><path d="M18.5 13h-13L2 9.5 5.5 6h13L22 9.5Z"/><path d="M12 13v8"/>',
    "dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "ban": '<circle cx="12" cy="12" r="10"/><path d="m4.9 4.9 14.2 14.2"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "arrow-circle": '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="m12 16 4-4-4-4"/>',
    "octagon": '<path d="M7.86 2h8.28L22 7.86v8.28L16.14 22H7.86L2 16.14V7.86z"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "circle-help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "scan": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "bar-chart": '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    "table": '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>',
    "eye": '<path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0"/><circle cx="12" cy="12" r="3"/>',
    "list": '<path d="M3 12h.01"/><path d="M3 18h.01"/><path d="M3 6h.01"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M8 6h13"/>',
    "award": '<path d="m15.477 12.89 1.515 8.526a.5.5 0 0 1-.81.47l-3.58-2.687a1 1 0 0 0-1.197 0l-3.586 2.686a.5.5 0 0 1-.81-.469l1.514-8.526"/><circle cx="12" cy="8" r="6"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor", stroke: float = 2) -> str:
    """Inline SVG icon (single line, safe inside html())."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="flex-shrink:0;vertical-align:middle;">{ICON_PATHS[name]}</svg>'
    )


def icon_tile(name: str, color: str, size: int = 40) -> str:
    """Rounded square with a tinted background and a coloured icon."""
    return (
        f'<span class="icon-tile" style="width:{size}px;height:{size}px;'
        f'background-color:{color}1F;">'
        f'{icon(name, int(size * 0.5), color)}</span>'
    )


# =========================================================
# CLASS NAMES - GTSRB
# =========================================================

CLASS_NAMES = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h",
    5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h",
    7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons",
}


# =========================================================
# SIGN FAMILIES (GTSRB groups, colours borrowed from real signs)
# =========================================================

FAMILIES = {
    "prohibition": {
        "label": "Prohibition",
        "icon": "ban",
        "color": "#C8102E",
        "ids": {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17},
    },
    "warning": {
        "label": "Warning",
        "icon": "alert",
        "color": "#E09F00",
        "ids": {11} | set(range(18, 32)),
    },
    "mandatory": {
        "label": "Mandatory",
        "icon": "arrow-circle",
        "color": "#0057B8",
        "ids": set(range(33, 41)),
    },
    "priority": {
        "label": "Priority / Stop",
        "icon": "octagon",
        "color": "#0F766E",
        "ids": {12, 13, 14},
    },
    "end": {
        "label": "End of restriction",
        "icon": "check-circle",
        "color": "#6B7280",
        "ids": {6, 32, 41, 42},
    },
}

FAMILY_COLORS = {f["label"]: f["color"] for f in FAMILIES.values()}


def get_family(class_id: int) -> dict:
    for family in FAMILIES.values():
        if class_id in family["ids"]:
            return family
    return {"label": "Other", "icon": "circle-help", "color": "#6B7280", "ids": set()}


def confidence_level(confidence: float) -> tuple:
    """Return (label, colour, icon) for a confidence value in %."""
    if confidence >= 90:
        return "High confidence", "#1E8E3E", "shield-check"
    if confidence >= 70:
        return "Medium confidence", "#E09F00", "gauge"
    return "Low confidence", "#C8102E", "alert"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Traffic Sign AI",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS (works in light and dark themes)
# =========================================================

st.markdown(
    """
    <style>

    :root {
        --radius: 14px;
        --radius-sm: 10px;
        --line: rgba(128, 128, 128, 0.22);
        --soft: rgba(128, 128, 128, 0.07);
        --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 6px 18px rgba(0, 0, 0, 0.06);
    }

    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* Generic icon tile */
    .icon-tile {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: var(--radius-sm);
        flex-shrink: 0;
    }

    /* Page header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 26px;
    }

    .page-header h1 {
        font-size: 32px;
        font-weight: 750;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 0 0 2px 0;
        padding: 0;
    }

    .page-header p {
        font-size: 15px;
        opacity: 0.62;
        margin: 0;
    }

    /* KPI cards */
    .kpi {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        padding: 16px 18px;
        border-radius: var(--radius);
        border: 1px solid var(--line);
        background-color: var(--soft);
        height: 100%;
        box-shadow: var(--shadow);
    }

    .kpi-body {
        min-width: 0;
    }

    .kpi-label {
        font-size: 13px;
        opacity: 0.68;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.01em;
    }

    .kpi-sub {
        font-size: 12.5px;
        opacity: 0.6;
        margin-top: 5px;
    }

    /* Result card: coloured strip = sign family */
    .result-card {
        padding: 24px 26px;
        border-radius: var(--radius);
        border: 1px solid var(--line);
        border-left-width: 8px;
        background-color: var(--soft);
        box-shadow: var(--shadow);
        margin-bottom: 14px;
    }

    .family-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 12px 4px 9px;
        border-radius: 999px;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .result-name {
        font-size: 28px;
        font-weight: 750;
        line-height: 1.2;
        letter-spacing: -0.01em;
        margin-bottom: 16px;
    }

    .result-row {
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    }

    .result-confidence {
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
    }

    .result-level {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 15px;
        font-weight: 600;
    }

    .result-meta {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        font-size: 13.5px;
        opacity: 0.68;
        margin-top: 14px;
    }

    .result-meta span {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .bar-track {
        height: 10px;
        border-radius: 999px;
        background-color: rgba(128, 128, 128, 0.2);
        overflow: hidden;
        margin-top: 16px;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* Section titles */
    .section-title {
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 17px;
        font-weight: 700;
        margin: 6px 0 12px 0;
    }

    .section-title svg {
        opacity: 0.75;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 4px 16px 4px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 10px;
    }

    .brand-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0057B8, #4C9AFF);
    }

    .brand-title {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .brand-sub {
        font-size: 12px;
        opacity: 0.6;
        margin-top: 2px;
    }

    .nav-label {
        font-size: 12.5px;
        font-weight: 600;
        opacity: 0.55;
        margin: 20px 0 6px 2px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start;
        text-align: left;
        border-radius: var(--radius-sm);
        padding: 0.55rem 0.9rem;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
        border-color: transparent;
        background-color: transparent;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(128, 128, 128, 0.15);
        border-color: transparent;
    }

    .st-key-theme_toggle button {
        border: 1px solid rgba(128, 128, 128, 0.35) !important;
        justify-content: center !important;
    }

    .side-stats {
        display: flex;
        gap: 10px;
        margin-top: 22px;
    }

    .side-stat {
        flex: 1;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--line);
        background-color: var(--soft);
    }

    .side-stat-value {
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
    }

    .side-stat-label {
        font-size: 12px;
        opacity: 0.6;
        margin-top: 3px;
    }

    /* Tabs, expanders, uploader: rounder, calmer */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: var(--radius);
        border: 1.5px dashed rgba(128, 128, 128, 0.45);
        background-color: var(--soft);
    }

    div[data-testid="stExpander"] details {
        border-radius: var(--radius-sm);
        border-color: var(--line);
    }

    div[data-testid="stImage"] img {
        border-radius: var(--radius-sm);
    }

    /* ---- Web-app chrome: hide Streamlit UI ---- */
    #MainMenu, footer, .stAppDeployButton {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .stApp {
        background-image:
            radial-gradient(900px 380px at 85% -5%, rgba(0, 87, 184, 0.12), transparent 60%),
            radial-gradient(700px 320px at 0% 0%, rgba(15, 118, 110, 0.08), transparent 60%);
    }

    /* ---- Top navigation bar ---- */
    .st-key-topnav {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 26px;
        background-color: var(--soft);
        backdrop-filter: blur(8px);
        box-shadow: var(--shadow);
    }

    .st-key-topnav .brand {
        border-bottom: none;
        margin: 0;
        padding: 0;
    }

    .st-key-topnav .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        padding: 0.45rem 0.9rem;
        white-space: nowrap;
    }

    .st-key-topnav .stButton > button[kind="secondary"],
    .st-key-topnav .stButton > button[data-testid="stBaseButton-secondary"] {
        border-color: transparent;
        background-color: transparent;
    }

    .st-key-topnav .stButton > button[kind="secondary"]:hover,
    .st-key-topnav .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(128, 128, 128, 0.15);
    }

    /* ---- Hero (empty dashboard) ---- */
    .hero {
        display: flex;
        align-items: center;
        gap: 28px;
        padding: 34px 36px;
        border-radius: 20px;
        color: #ffffff;
        background: linear-gradient(120deg, #003A7A 0%, #0057B8 55%, #0F766E 100%);
        box-shadow: 0 10px 30px rgba(0, 87, 184, 0.25);
        margin-bottom: 18px;
        animation: rise 0.5s ease both;
    }

    .hero h2 {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 0 0 8px 0;
        padding: 0;
        color: #ffffff;
    }

    .hero p {
        font-size: 16px;
        opacity: 0.88;
        margin: 0;
        max-width: 560px;
    }

    .hero-chips {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        font-weight: 600;
        padding: 5px 12px 5px 9px;
        border-radius: 999px;
        background-color: rgba(255, 255, 255, 0.16);
    }

    .hero-art {
        flex-shrink: 0;
        width: 110px;
        height: 110px;
        border-radius: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(255, 255, 255, 0.14);
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: none; }
    }

    @media (prefers-reduced-motion: reduce) {
        .hero { animation: none; }
    }

    @media (max-width: 768px) {
        .hero { flex-direction: column; align-items: flex-start; padding: 24px; }
        .hero-art { display: none; }
        .page-header h1 { font-size: 26px; }
    }

    /* Footer */
    .footer {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        opacity: 0.55;
        font-size: 13.5px;
        padding-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            f"Model not found:\n{MODEL_PATH}",
            icon=":material/error:",
        )

        st.stop()

    return keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "logged" not in st.session_state:
    st.session_state.logged = set()

# (name, material icon for native widgets, svg icon for headers)
PAGES = ["Dashboard", "Classifier", "Batch analysis", "History", "Model"]

PAGE_ICONS = {
    "Dashboard": ("dashboard", "dashboard"),
    "Classifier": ("image_search", "search"),
    "Batch analysis": ("library_add_check", "layers"),
    "History": ("history", "clock"),
    "Model": ("psychology", "cpu"),
}

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]


def go_to(page_name: str):
    st.session_state.page = page_name


# ---------------------------------------------------------
# THEME (light / dark)
# ---------------------------------------------------------

THEMES = {
    "dark": {
        "theme.base": "dark",
        "theme.primaryColor": "#4C9AFF",
        "theme.backgroundColor": "#0E1117",
        "theme.secondaryBackgroundColor": "#1B1F2A",
        "theme.textColor": "#FAFAFA",
    },
    "light": {
        "theme.base": "light",
        "theme.primaryColor": "#0057B8",
        "theme.backgroundColor": "#FFFFFF",
        "theme.secondaryBackgroundColor": "#F0F2F6",
        "theme.textColor": "#1F2937",
    },
}


def detect_theme() -> str:
    """Current theme of the browser session (falls back to dark)."""
    try:
        detected = st.context.theme.type
    except Exception:
        detected = None

    return detected if detected in ("light", "dark") else "dark"


if "theme" not in st.session_state:
    st.session_state.theme = detect_theme()


def toggle_theme():
    new_theme = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme = new_theme

    try:
        for option, value in THEMES[new_theme].items():
            st._config.set_option(option, value)
    except Exception:
        pass


# =========================================================
# HELPERS
# =========================================================

def html(text: str) -> str:
    """Collapse an indented HTML block into one line (safe for st.markdown)."""
    return "".join(line.strip() for line in text.splitlines())


def to_data_uri(image: Image.Image, size: int = 48) -> str:
    thumb = image.copy()
    thumb.thumbnail((size, size))

    buffer = io.BytesIO()
    thumb.save(buffer, format="PNG")

    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()


def preprocess_batch(images: list) -> np.ndarray:
    """Same preprocessing as training: RGB, 32x32, scaled to [0, 1]."""
    arrays = [
        np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        for img in images
    ]
    return np.stack(arrays, axis=0)


def predict_batch(images: list) -> tuple:
    """Return (probabilities [n, 43], inference ms per image)."""
    batch = preprocess_batch(images)

    start = time.perf_counter()
    probabilities = model.predict(batch, verbose=0, batch_size=64)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return probabilities, elapsed_ms / len(images)


def make_record(name: str, image: Image.Image, probs, ms: float) -> dict:
    top_class = int(np.argmax(probs))
    family = get_family(top_class)

    return {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "File": name,
        "Class ID": top_class,
        "Sign": CLASS_NAMES.get(top_class, f"Class {top_class}"),
        "Family": family["label"],
        "Confidence": round(float(probs[top_class] * 100), 2),
        "Inference (ms)": round(ms, 1),
        "Preview": to_data_uri(image),
    }


def log_record(record: dict, key: str) -> bool:
    if key in st.session_state.logged:
        return False

    st.session_state.logged.add(key)
    st.session_state.history.append(record)
    return True


def history_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.history)


def kpi(
    label: str,
    value: str,
    sub: str = "",
    color: str = "#0057B8",
    icon_name: str = "activity",
) -> str:
    return html(
        f"""
        <div class="kpi">
            {icon_tile(icon_name, color, 42)}
            <div class="kpi-body">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
        </div>
        """
    )


def render_bar(percent: float, color: str) -> str:
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{percent:.2f}%; '
        f'background-color:{color};"></div></div>'
    )


def page_header(title: str, subtitle: str):
    icon_name = PAGE_ICONS[title][1] if title in PAGE_ICONS else "signpost"

    st.markdown(
        html(
            f"""
            <div class="page-header">
                {icon_tile(icon_name, "#0057B8", 52)}
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def section_title(text: str, icon_name: str = None):
    glyph = icon(icon_name, 18) if icon_name else ""

    st.markdown(
        f'<div class="section-title">{glyph}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def family_chart(df: pd.DataFrame, count_col: str = "Count"):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Family:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{count_col}:Q", title=None),
            color=alt.Color(
                "Family:N",
                scale=alt.Scale(
                    domain=list(FAMILY_COLORS.keys()),
                    range=list(FAMILY_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=["Family", count_col],
        )
        .properties(height=260)
    )


# =========================================================
# SIDEBAR (navigation + settings)
# =========================================================

with st.sidebar:

    st.markdown('<div class="nav-label">Settings</div>', unsafe_allow_html=True)

    with st.expander("Analysis settings", icon=":material/tune:"):

        top_k = st.slider(
            "Predictions to show",
            min_value=3,
            max_value=10,
            value=5,
        )

        threshold = st.slider(
            "Reliability threshold (%)",
            min_value=0,
            max_value=100,
            value=70,
            help="Analyses below this confidence are flagged for review.",
        )

    st.markdown('<div class="nav-label">Appearance</div>', unsafe_allow_html=True)

    is_dark = st.session_state.theme == "dark"

    st.button(
        (
            ":material/light_mode:  Switch to light mode"
            if is_dark
            else ":material/dark_mode:  Switch to dark mode"
        ),
        key="theme_toggle",
        on_click=toggle_theme,
        use_container_width=True,
    )

    # Filled at the end of the script, once the current analysis is logged
    side_stats_slot = st.empty()


with st.container(key="topnav"):

    brand_col, *nav_cols = st.columns(
        [2.4, 1.15, 1.1, 1.5, 1.05, 1.0],
        vertical_alignment="center",
    )

    with brand_col:
        st.markdown(
            html(
                f"""
                <div class="brand">
                    <span class="brand-logo">{icon("signpost", 22, "#ffffff")}</span>
                    <div>
                        <div class="brand-title">Traffic Sign AI</div>
                        <div class="brand-sub">Recognition of 43 GTSRB signs</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    for nav_col, page_name in zip(nav_cols, PAGES):
        with nav_col:
            st.button(
                f":material/{PAGE_ICONS[page_name][0]}: {page_name}",
                key=f"nav_{page_name}",
                on_click=go_to,
                args=(page_name,),
                type="primary" if st.session_state.page == page_name else "secondary",
                use_container_width=True,
            )

page = st.session_state.page


# =========================================================
# PAGE 1 - DASHBOARD
# =========================================================

if page == PAGES[0]:

    page_header(
        "Dashboard",
        "Overview of the analyses made during this session.",
    )

    df = history_df()

    if df.empty:

        st.markdown(
            html(
                f"""
                <div class="hero">
                    <div>
                        <h2>Identify any traffic sign in a second</h2>
                        <p>Upload a photo or use your camera. The CNN returns the sign,
                        its family and how confident it is. Every analysis lands here.</p>
                        <div class="hero-chips">
                            <span class="hero-chip"><span class="material-symbols-rounded">target</span></span>
                            <span class="hero-chip">{icon("signpost", 15, "#ffffff")} 43 sign classes</span>
                            <span class="hero-chip">{icon("clock", 15, "#ffffff")} Under a second per image</span>
                        </div>
                    </div>
                    <div class="hero-art">{icon("signpost", 56, "#ffffff", 1.6)}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.button(
                ":material/image_search:  Classify a sign",
                on_click=go_to,
                args=(PAGES[1],),
                use_container_width=True,
            )

        with col_b:
            st.button(
                ":material/library_add_check:  Analyse several images",
                on_click=go_to,
                args=(PAGES[2],),
                use_container_width=True,
            )

    else:

        total = len(df)
        avg_conf = df["Confidence"].mean()
        reliable = int((df["Confidence"] >= threshold).sum())
        to_review = total - reliable
        avg_ms = df["Inference (ms)"].mean()
        distinct = df["Class ID"].nunique()

        avg_label, avg_color, avg_icon = confidence_level(avg_conf)

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.markdown(
                kpi("Analyses", f"{total}", "this session", "#0057B8", "activity"),
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                kpi("Average confidence", f"{avg_conf:.1f}%", avg_label, avg_color, avg_icon),
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                kpi(
                    "Reliable",
                    f"{reliable}",
                    f"≥ {threshold}% confidence",
                    "#1E8E3E",
                    "check-circle",
                ),
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                kpi(
                    "To review",
                    f"{to_review}",
                    f"< {threshold}% confidence",
                    "#C8102E" if to_review else "#6B7280",
                    "flag",
                ),
                unsafe_allow_html=True,
            )

        with k5:
            st.markdown(
                kpi(
                    "Distinct signs",
                    f"{distinct}",
                    f"{avg_ms:.0f} ms / image",
                    "#0F766E",
                    "signpost",
                ),
                unsafe_allow_html=True,
            )

        st.write("")

        left, right = st.columns(2, gap="large")

        with left:

            section_title("Analyses by sign family", "bar-chart")

            family_df = (
                df.groupby("Family")
                .size()
                .reset_index(name="Count")
            )

            st.altair_chart(
                family_chart(family_df),
                use_container_width=True,
            )

        with right:

            section_title("Confidence over time", "trending-up")

            trend_df = pd.DataFrame(
                {
                    "Analysis": range(1, total + 1),
                    "Confidence": df["Confidence"].values,
                }
            )

            line = (
                alt.Chart(trend_df)
                .mark_line(point=True, color="#0057B8")
                .encode(
                    x=alt.X("Analysis:Q", title="Analysis #", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y(
                        "Confidence:Q",
                        title="Confidence (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    tooltip=["Analysis", "Confidence"],
                )
            )

            rule = (
                alt.Chart(pd.DataFrame({"y": [threshold]}))
                .mark_rule(strokeDash=[5, 5], color="#C8102E")
                .encode(y="y:Q")
            )

            st.altair_chart(
                (line + rule).properties(height=260),
                use_container_width=True,
            )

            st.caption(f"Red dashed line = reliability threshold ({threshold}%).")

        st.write("")

        bottom_left, bottom_right = st.columns([1, 2], gap="large")

        with bottom_left:

            section_title("Most detected signs", "award")

            top_signs = (
                df["Sign"]
                .value_counts()
                .head(5)
                .rename_axis("Sign")
                .reset_index(name="Detections")
            )

            st.dataframe(
                top_signs,
                hide_index=True,
                use_container_width=True,
            )

        with bottom_right:

            section_title("Recent analyses", "clock")

            recent = df.tail(8).iloc[::-1]

            st.dataframe(
                recent[["Preview", "File", "Sign", "Confidence", "Time"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Preview": st.column_config.ImageColumn("Preview"),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )


# =========================================================
# PAGE 2 - CLASSIFIER
# =========================================================

elif page == PAGES[1]:

    page_header(
        "Classifier",
        "Upload a photo or use the camera to identify a traffic sign.",
    )

    tab_upload, tab_camera = st.tabs(
        [":material/upload: Upload an image", ":material/photo_camera: Use the camera"]
    )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with tab_camera:
        camera_file = st.camera_input(
            "Take a picture",
            label_visibility="collapsed",
        )

    source_file = uploaded_file if uploaded_file is not None else camera_file

    if source_file is None:

        st.info(
            "Upload an image or take a photo of a traffic sign to start. "
            "Each analysis is added to the dashboard.",
            icon=":material/info:",
        )

    else:

        raw_bytes = source_file.getvalue()
        file_key = hashlib.md5(raw_bytes).hexdigest()

        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        probabilities_batch, inference_ms = predict_batch([image])
        probabilities = probabilities_batch[0]

        file_name = getattr(source_file, "name", "camera.png")
        record = make_record(file_name, image, probabilities, inference_ms)
        log_record(record, file_key)

        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        predicted_class = int(top_indices[0])
        confidence = float(probabilities[predicted_class] * 100)
        predicted_name = CLASS_NAMES.get(
            predicted_class, f"Class {predicted_class}"
        )

        family = get_family(predicted_class)
        level_label, level_color, level_icon = confidence_level(confidence)

        st.markdown("---")

        left_column, right_column = st.columns([1, 1], gap="large")

        with left_column:

            section_title("Input image", "image")

            st.image(image, use_container_width=True)

            with st.expander("What the model actually sees", icon=":material/visibility:"):

                model_view = image.resize(
                    (IMG_SIZE, IMG_SIZE)
                ).resize((256, 256), Image.NEAREST)

                st.image(
                    model_view,
                    caption=f"Resized to {IMG_SIZE} × {IMG_SIZE} before prediction",
                )

        with right_column:

            section_title("Prediction", "target")

            st.markdown(
                html(
                    f"""
                    <div class="result-card" style="border-left-color:{family['color']};">
                        <span class="family-badge" style="background-color:{family['color']};">
                            {icon(family['icon'], 15, '#ffffff')}
                            {family['label']}
                        </span>
                        <div class="result-name">{predicted_name}</div>
                        <div class="result-row">
                            <span class="result-confidence" style="color:{level_color};">
                                {confidence:.2f}%
                            </span>
                            <span class="result-level" style="color:{level_color};">
                                {icon(level_icon, 18, level_color)}
                                {level_label}
                            </span>
                        </div>
                        {render_bar(confidence, level_color)}
                        <div class="result-meta">
                            <span>{icon('database', 15)} Class ID {predicted_class}</span>
                            <span>{icon('clock', 15)} Inference {inference_ms:.0f} ms</span>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            if confidence < threshold:

                second_class = int(top_indices[1])

                st.warning(
                    f"The confidence is below {threshold}%. "
                    f"The model also considers "
                    f"**{CLASS_NAMES.get(second_class, f'Class {second_class}')}** "
                    f"({probabilities[second_class] * 100:.2f}%). "
                    "Try a sharper, closer photo of the sign.",
                    icon=":material/warning:",
                )

        st.markdown("---")

        results_df = pd.DataFrame(
            {
                "Rank": range(1, len(top_indices) + 1),
                "Class ID": [int(i) for i in top_indices],
                "Sign": [
                    CLASS_NAMES.get(int(i), f"Class {int(i)}")
                    for i in top_indices
                ],
                "Family": [get_family(int(i))["label"] for i in top_indices],
                "Confidence": [
                    round(float(probabilities[int(i)] * 100), 2)
                    for i in top_indices
                ],
            }
        )

        tab_chart, tab_table = st.tabs(
            [
                f":material/bar_chart: Top {top_k} predictions",
                ":material/table_chart: Table & export",
            ]
        )

        with tab_chart:

            bars = (
                alt.Chart(results_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X(
                        "Confidence:Q",
                        title="Confidence (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    y=alt.Y("Sign:N", sort="-x", title=None),
                    color=alt.Color(
                        "Family:N",
                        scale=alt.Scale(
                            domain=list(FAMILY_COLORS.keys()),
                            range=list(FAMILY_COLORS.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=["Rank", "Sign", "Family", "Confidence"],
                )
                .properties(height=max(180, 42 * len(results_df)))
            )

            st.altair_chart(bars, use_container_width=True)

        with tab_table:

            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

            st.download_button(
                ":material/download: Download results (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_sign_prediction.csv",
                mime="text/csv",
            )


# =========================================================
# PAGE 3 - BATCH ANALYSIS
# =========================================================

elif page == PAGES[2]:

    page_header(
        "Batch analysis",
        "Classify many images at once and review the ones that need attention.",
    )

    files = st.file_uploader(
        "Choose several images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not files:

        st.info(
            "Upload several traffic sign images to analyse them together.",
            icon=":material/info:",
        )

    else:

        raw_list = [(f.name, f.getvalue()) for f in files]
        signature = tuple(hashlib.md5(raw).hexdigest() for _, raw in raw_list)

        if st.session_state.get("batch_signature") != signature:

            images = []
            names = []
            keys = []

            for (name, raw), key in zip(raw_list, signature):
                try:
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    names.append(name)
                    keys.append(key)
                except Exception:
                    st.warning(f"Could not read {name}, skipped.", icon=":material/warning:")

            with st.spinner(f"Analysing {len(images)} image(s)..."):
                probs_all, ms_per_image = predict_batch(images)

            records = [
                make_record(n, img, p, ms_per_image)
                for n, img, p in zip(names, images, probs_all)
            ]

            for record, key in zip(records, keys):
                log_record(record, key)

            st.session_state.batch_signature = signature
            st.session_state.batch_records = records

        records = st.session_state.batch_records
        batch_df = pd.DataFrame(records)

        total = len(batch_df)
        avg_conf = batch_df["Confidence"].mean()
        to_review = int((batch_df["Confidence"] < threshold).sum())

        avg_label, avg_color, avg_icon = confidence_level(avg_conf)

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(
                kpi("Images analysed", f"{total}", "in this batch", "#0057B8", "layers"),
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                kpi("Average confidence", f"{avg_conf:.1f}%", avg_label, avg_color, avg_icon),
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                kpi(
                    "Reliable",
                    f"{total - to_review}",
                    f"≥ {threshold}% confidence",
                    "#1E8E3E",
                    "check-circle",
                ),
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                kpi(
                    "To review",
                    f"{to_review}",
                    f"< {threshold}% confidence",
                    "#C8102E" if to_review else "#6B7280",
                    "flag",
                ),
                unsafe_allow_html=True,
            )

        st.write("")

        left, right = st.columns([2, 1], gap="large")

        with left:

            section_title("Results", "list")

            show_df = batch_df.copy()
            show_df["Status"] = np.where(
                show_df["Confidence"] >= threshold, "Reliable", "Review"
            )

            st.dataframe(
                show_df[
                    ["Preview", "File", "Sign", "Family", "Confidence", "Status"]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Preview": st.column_config.ImageColumn("Preview"),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

            st.download_button(
                ":material/download: Download batch results (CSV)",
                data=show_df.drop(columns=["Preview"])
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="traffic_sign_batch_results.csv",
                mime="text/csv",
            )

        with right:

            section_title("Family distribution", "bar-chart")

            batch_family = (
                batch_df.groupby("Family").size().reset_index(name="Count")
            )

            st.altair_chart(
                family_chart(batch_family),
                use_container_width=True,
            )


# =========================================================
# PAGE 4 - HISTORY
# =========================================================

elif page == PAGES[3]:

    page_header(
        "History",
        "All analyses of this session, with filters and export.",
    )

    df = history_df()

    if df.empty:

        st.info(
            "The history is empty. Classify a sign to start.",
            icon=":material/info:",
        )

    else:

        filter_left, filter_right = st.columns([2, 1], gap="large")

        with filter_left:
            selected_families = st.multiselect(
                "Sign family",
                options=list(FAMILY_COLORS.keys()),
                default=list(FAMILY_COLORS.keys()),
            )

        with filter_right:
            conf_min, conf_max = st.slider(
                "Confidence range (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
            )

        filtered = df[
            df["Family"].isin(selected_families)
            & df["Confidence"].between(conf_min, conf_max)
        ]

        st.caption(f"{len(filtered)} of {len(df)} analyses shown.")

        st.dataframe(
            filtered.iloc[::-1][
                [
                    "Preview",
                    "Time",
                    "File",
                    "Class ID",
                    "Sign",
                    "Family",
                    "Confidence",
                    "Inference (ms)",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Preview": st.column_config.ImageColumn("Preview"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

        action_left, action_right = st.columns(2)

        with action_left:
            st.download_button(
                ":material/download: Download history (CSV)",
                data=filtered.drop(columns=["Preview"])
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="traffic_sign_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with action_right:
            if st.button(":material/delete: Clear history", use_container_width=True):
                st.session_state.history = []
                st.session_state.logged = set()
                st.session_state.pop("batch_signature", None)
                st.session_state.pop("batch_records", None)
                st.rerun()


# =========================================================
# PAGE 5 - MODEL
# =========================================================

elif page == PAGES[4]:

    page_header(
        "Model",
        "Technical details of the CNN and the traffic sign classes.",
    )

    try:
        params = f"{model.count_params():,}".replace(",", " ")
    except Exception:
        params = "—"

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            kpi("Classes", "43", "GTSRB", "#0057B8", "signpost"),
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            kpi("Input size", f"{IMG_SIZE} × {IMG_SIZE}", "RGB, scaled to [0, 1]", "#0F766E", "scan"),
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            kpi("Test accuracy", TEST_ACCURACY, "GTSRB test set", "#1E8E3E", "target"),
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            kpi("Parameters", params, "trainable + frozen", "#E09F00", "cpu"),
            unsafe_allow_html=True,
        )

    st.write("")

    section_title("Traffic sign classes", "list")

    search = st.text_input(
        "Search a sign",
        placeholder="Search a sign, e.g. speed, stop, curve...",
        label_visibility="collapsed",
        icon=":material/search:",
    )

    classes_df = pd.DataFrame(
        {
            "Class ID": list(CLASS_NAMES.keys()),
            "Sign": list(CLASS_NAMES.values()),
            "Family": [get_family(i)["label"] for i in CLASS_NAMES],
        }
    )

    if search:
        classes_df = classes_df[
            classes_df["Sign"].str.contains(search, case=False, na=False)
            | classes_df["Family"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(
        classes_df,
        hide_index=True,
        use_container_width=True,
        height=420,
    )

    st.caption(
        "Test accuracy was measured on the GTSRB test set. Photos taken in "
        "real conditions (blur, glare, partial occlusion) may give lower confidence."
    )


# =========================================================
# FOOTER
# =========================================================

side_stats_slot.markdown(
    html(
        f"""
        <div class="side-stats">
            <div class="side-stat">
                <div class="side-stat-value">{TEST_ACCURACY}</div>
                <div class="side-stat-label">Test accuracy</div>
            </div>
            <div class="side-stat">
                <div class="side-stat-value">{len(st.session_state.history)}</div>
                <div class="side-stat-label">Analyses</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="footer">
        {icon("signpost", 16)}
        <span>Traffic Sign Classification • CNN + TensorFlow/Keras + Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)