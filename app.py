import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import time
import json
import hashlib

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { padding: 1rem; border-radius: 12px; margin-bottom: 0.5rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stSelectbox, .stMultiSelect, .stNumberInput, .stRadio { margin-top: -10px; }
    
    /* Login Form Styling */
    .auth-container {
        border: 1px solid #ddd;
        padding: 30px;
        border-radius: 15px;
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_FILE = "users.json"


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_user(username, password):
    users = load_users()
    users[username] = hash_password(password)
    with open(DB_FILE, "w") as f:
        json.dump(users, f)


def authenticate(username, password):
    users = load_users()
    if username in users:
        return users[username] == hash_password(password)
    return False


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.logged_in:
    if not os.path.exists("disease_predictor.h5"):
        st.error("Model file missing.")
        st.stop()

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("disease_predictor.h5")

    @st.cache_data
    def load_assets():
        try:
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
            with open("symptoms_list.pkl", "rb") as f:
                sym_list = pickle.load(f)
            desc_df = pd.read_csv("symptom_Description.csv")
            prec_df = pd.read_csv("symptom_precaution.csv")
            desc_dict = dict(zip(desc_df.iloc[:, 0], desc_df.iloc[:, 1]))
            prec_dict = {}
            for _, row in prec_df.iterrows():
                prec_dict[row.iloc[0]] = [x for x in row.iloc[1:] if pd.notna(x)]
            return le, sym_list, desc_dict, prec_dict
        except Exception as e:
            st.error(f"Error loading assets: {e}")
            st.stop()

    model = load_model()
    label_encoder, all_symptoms, desc_dict, prec_dict = load_assets()
    display_symptoms = {s.replace("_", " ").title(): s for s in all_symptoms}

    CRITICAL_DISEASES = [
        "Heart attack",
        "Paralysis (brain hemorrhage)",
        "Alcoholic hepatitis",
        "Hepatitis C",
        "Hepatitis D",
        "Hepatitis E",
        "Typhoid",
        "Tuberculosis",
        "Pneumonia",
        "Dengue",
        "Chicken pox",
    ]


def add_message(role, content, is_html=False):
    st.session_state.history.append(
        {"role": role, "content": content, "is_html": is_html}
    )


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.history = []
    st.session_state.step = 0
    st.rerun()


def auth_page():
    st.markdown(
        "<h1 style='text-align: center; color: #2C3E50;'>🔐 MediChat Access</h1>",
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login ➤", use_container_width=True):
                if authenticate(user, pw):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.success("Success!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("signup_form"):
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            if st.form_submit_button("Create Account", use_container_width=True):
                if new_u in load_users():
                    st.warning("User exists!")
                else:
                    save_user(new_u, new_p)
                    st.success("Created! Please login.")


def health_bot_page():
    with st.sidebar:
        st.write(f"👤 **{st.session_state.username}**")
        if st.button("Logout"):
            logout()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "inputs" not in st.session_state:
        st.session_state.inputs = {}

    st.markdown(
        "<h2 style='text-align: center; color: #2C3E50;'>🤖 AI Health Assistant</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align: center; color: #999; font-size: 0.8em; margin-bottom: 20px;'>Virtual Triage System</div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg.get("is_html"):
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.write(msg["content"])

    if st.session_state.step == 0:
        intro = f"Hello {st.session_state.username}! I need to gather some basic details to understand your health context better. Ready?"
        add_message("assistant", intro)
        st.session_state.step = 1
        st.rerun()

    # --- STEP 1: GENDER & AGE ---
    elif st.session_state.step == 1:
        # Check if we already asked (prevents duplicate messages on rerun)
        if (
            not st.session_state.history
            or st.session_state.history[-1]["role"] != "assistant"
        ):
            add_message(
                "assistant",
                "Let's start with the basics. What is your **Gender** and **Age**?",
            )
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gen")
        with c2:
            age = st.number_input("Age", 1, 120, 25, key="age")

        if st.button("Next ➤"):
            add_message("user", f"I am a {age}-year-old {gender}.")
            st.session_state.inputs.update({"age": age, "gender": gender})
            st.session_state.step = 2
            st.rerun()

    # --- STEP 2: PHYSICAL STATS (Height/Weight) ---
    elif st.session_state.step == 2:
        if st.session_state.history[-1]["role"] != "assistant":
            add_message(
                "assistant", "Thanks. Could you share your **Height** and **Weight**?"
            )
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            h = st.number_input("Height (cm)", 50, 250, 170, key="h")
        with c2:
            w = st.number_input("Weight (kg)", 20, 200, 70, key="w")

        if st.button("Next ➤"):
            add_message("user", f"Height: {h}cm, Weight: {w}kg.")
            st.session_state.inputs.update({"height": h, "weight": w})
            st.session_state.step = 3
            st.rerun()

    # --- STEP 3: LIFESTYLE & RISKS ---
    elif st.session_state.step == 3:
        if st.session_state.history[-1]["role"] != "assistant":
            add_message(
                "assistant",
                "Do you have any **existing conditions** or habits (Smoking/Alcohol)? Also, how is your **sleep**?",
            )
            st.rerun()

        risks = st.multiselect(
            "Conditions/Habits:",
            [
                "Diabetes",
                "Hypertension",
                "Asthma",
                "Heart Condition",
                "Smoker",
                "Alcohol Consumer",
                "None",
            ],
            key="risk",
        )
        sleep = st.selectbox(
            "Avg Sleep:",
            ["< 5 hrs", "5-7 hrs", "7-9 hrs", "> 9 hrs"],
            index=2,
            key="sleep",
        )

        if st.button("Next ➤"):
            r_str = ", ".join(risks) if risks else "None"
            add_message("user", f"Risks: {r_str}. Sleep: {sleep}.")
            st.session_state.inputs.update({"risks": risks, "sleep": sleep})
            st.session_state.step = 4
            st.rerun()

    # --- STEP 4: SYMPTOMS ---
    elif st.session_state.step == 4:
        if st.session_state.history[-1]["role"] != "assistant":
            add_message(
                "assistant",
                "Profile updated. Now, **what symptoms are you experiencing today?**",
            )
            st.rerun()

        syms = st.multiselect(
            "Search symptoms:",
            list(display_symptoms.keys()),
            placeholder="Type to search...",
            key="sym",
        )

        if st.button("Confirm Symptoms ➤"):
            if not syms:
                st.warning("Please select at least one symptom.")
            else:
                add_message("user", f"I am feeling: {', '.join(syms)}")
                st.session_state.inputs["symptoms"] = syms
                st.session_state.step = 5
                st.rerun()

    # --- STEP 5: DURATION ---
    elif st.session_state.step == 5:
        if st.session_state.history[-1]["role"] != "assistant":
            add_message(
                "assistant",
                "Understood. **How many days** have you had these symptoms?",
            )
            st.rerun()

        days = st.number_input("Days:", 1, 365, 1, key="days")
        if st.button("Analyze Now 🔍"):
            add_message("user", f"{days} days.")
            st.session_state.inputs["days"] = days
            st.session_state.step = 6
            st.rerun()

    # --- STEP 6: ANALYSIS ---
    elif st.session_state.step == 6:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                time.sleep(1)

                # PREDICTION
                user_syms = [
                    display_symptoms[s] for s in st.session_state.inputs["symptoms"]
                ]
                ivec = np.zeros(len(all_symptoms))
                for s in user_syms:
                    if s in all_symptoms:
                        ivec[all_symptoms.index(s)] = 1

                pred = model.predict(ivec.reshape(1, -1), verbose=0)[0]
                top_idx = np.argsort(pred)[-3:][::-1]
                disease = label_encoder.inverse_transform([top_idx[0]])[0]
                conf = pred[top_idx[0]] * 100

                # TRIAGE LOGIC
                risks = st.session_state.inputs.get("risks", [])
                days = st.session_state.inputs["days"]
                is_crit = disease in CRITICAL_DISEASES
                has_risk = any(
                    r in risks
                    for r in ["Diabetes", "Hypertension", "Heart Condition", "Smoker"]
                )

                if is_crit:
                    col, bg, icon, ti, sub = (
                        "#ff4b4b",
                        "#262730",
                        "🚨",
                        "Immediate Medical Attention Required",
                        "Condition is severe. Go to ER.",
                    )
                elif conf < 45:
                    col, bg, icon, ti, sub = (
                        "#ffa726",
                        "#262730",
                        "❓",
                        "Consult a Doctor",
                        "Symptoms ambiguous. Tests needed.",
                    )
                elif has_risk:
                    col, bg, icon, ti, sub = (
                        "#ffa726",
                        "#262730",
                        "⚠️",
                        "Consult a Doctor",
                        f"Caution due to history ({', '.join(risks)}).",
                    )
                elif days > 5:
                    col, bg, icon, ti, sub = (
                        "#ffa726",
                        "#262730",
                        "🕒",
                        "See a Doctor",
                        "Symptoms persisted too long.",
                    )
                else:
                    col, bg, icon, ti, sub = (
                        "#66bb6a",
                        "#262730",
                        "🏡",
                        "Home Care Sufficient",
                        "Monitor for 24 hours.",
                    )

                # HTML OUTPUT
                desc = desc_dict.get(disease, "No details.")
                pre = prec_dict.get(disease, [])
                pre_li = (
                    "".join(
                        [
                            f"<li>{p.replace('Asprin', 'Aspirin').title()}</li>"
                            for p in pre
                        ]
                    )
                    if pre
                    else "<li>Rest</li>"
                )

                html = f"""
                <style>@keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} 100% {{ opacity: 1; }} }} .crit {{ animation: {"pulse 1.5s infinite" if is_crit else "none"}; display:inline-block; }}</style>
                <div style="background:{bg}; padding:20px; border-radius:15px; border-left:6px solid {col}; margin-top:10px; box-shadow:0 4px 10px rgba(0,0,0,0.3);">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0; color:#fff;"><span class='crit'>{icon}</span> Diagnosis</h3>
                        <span style="background:{col}; color:#fff; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;">{conf:.1f}% Match</span>
                    </div>
                    <h2 style="color:{col}; margin:10px 0;">{disease}</h2>
                    <p style="color:#ccc; font-style:italic; font-size:0.9em;">{desc}</p>
                    <hr style="border-color:#444;">
                    <strong style="color:#eee;">Actions:</strong>
                    <ul style="color:#ddd; padding-left:20px;">{pre_li}</ul>
                    <div style="background:#333; padding:15px; border-radius:10px; margin-top:15px;">
                        <strong style="color:{col};">AI Advice:</strong><br>
                        <strong style="color:#fff;">{ti}</strong><br>
                        <span style="color:#ccc; font-size:0.9em;">{sub}</span>
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                add_message("assistant", html, is_html=True)

                # Alternatives
                if len(top_idx) > 1:
                    alts = []
                    for i in range(1, 3):
                        if pred[top_idx[i]] > 0.05:
                            d = label_encoder.inverse_transform([top_idx[i]])[0]
                            alts.append(f"**{d}** ({pred[top_idx[i]] * 100:.1f}%)")
                    if alts:
                        amsg = "💡 **Also considered:** " + ", ".join(alts)
                        st.write(amsg)
                        add_message("assistant", amsg)

                st.session_state.step = 7

    # --- STEP 7: RESET ---
    elif st.session_state.step == 7:
        if st.button("Start New Consultation 🔄"):
            st.session_state.history = []
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.rerun()

    st.markdown(
        "<div style='text-align:center; margin-top:50px; color:#666; font-size:0.8em;'>⚠️ <b>DISCLAIMER:</b> Not professional medical advice.</div>",
        unsafe_allow_html=True,
    )


if st.session_state.logged_in:
    health_bot_page()
else:
    auth_page()
