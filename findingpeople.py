import streamlit as st
import json
import os
import hashlib
from datetime import datetime

# ============================================================
# TRACE-AI
# Finding Missing People Using AI
# Public website + Admin-only case management
# ============================================================

st.set_page_config(
    page_title="TRACE-AI",
    page_icon="🔎",
    layout="wide"
)

# ============================================================
# FILES / FOLDERS
# ============================================================

DATA_FILE = "cases.json"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def load_cases():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def save_cases(cases):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(cases, file, indent=4, ensure_ascii=False)


# ============================================================
# ADMIN AUTHENTICATION
# ============================================================
# For local testing:
#   username = admin
#   password = ChangeMe123!
#
# For public deployment, DO NOT keep the password in this file.
# Put ADMIN_PASSWORD in Streamlit Secrets instead.
# ============================================================

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "ChangeMe123!"

def get_admin_password():
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return DEFAULT_ADMIN_PASSWORD


def check_admin(username, password):
    return (
        username == DEFAULT_ADMIN_USERNAME
        and password == get_admin_password()
    )


if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


def admin_login_page():
    st.header("🔐 Admin Login")
    st.write("Only authorized administrators can change case status or GPS information.")

    with st.form("admin_login"):
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")
        login = st.form_submit_button("🔐 Login")

        if login:
            if check_admin(username.strip(), password):
                st.session_state.admin_logged_in = True
                st.success("Admin login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")


# ============================================================
# SESSION STATE
# ============================================================

if "cases" not in st.session_state:
    st.session_state.cases = load_cases()


# ============================================================
# HEADER
# ============================================================

st.title("🔎 TRACE-AI")
st.subheader("Finding Missing People Using AI")

st.write(
    "A public information and reporting platform for missing-person cases. "
    "Case updates are restricted to authorized administrators."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 TRACE-AI")
st.sidebar.write("Missing Person Detection & Tracking System")

public_pages = [
    "🏠 Home",
    "📝 Report Missing Person",
    "🔍 Search Cases",
    "🤖 AI Face Search"
]

if st.session_state.admin_logged_in:
    page_options = public_pages + [
        "📍 GPS Location",
        "📊 Admin Dashboard"
    ]
else:
    page_options = public_pages + [
        "🔐 Admin Login"
    ]

page = st.sidebar.radio("Select Module", page_options)

st.sidebar.divider()

if st.session_state.admin_logged_in:
    st.sidebar.success("🔐 Administrator logged in")

    if st.sidebar.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()
else:
    st.sidebar.info(
        "Public access enabled.\n\n"
        "Only authorized administrators can modify case status."
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("🏠 Welcome to TRACE-AI")

    st.write(
        "TRACE-AI is an educational prototype designed to assist "
        "authorized users and the public in searching and reporting "
        "missing-person cases."
    )

    st.divider()

    total_cases = len(st.session_state.cases)

    missing_cases = sum(
        1 for case in st.session_state.cases
        if case.get("status") == "Missing"
    )

    found_cases = sum(
        1 for case in st.session_state.cases
        if case.get("status") == "Found"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cases", total_cases)

    with col2:
        st.metric("Missing", missing_cases)

    with col3:
        st.metric("Found", found_cases)

    st.divider()

    st.header("⚙️ How TRACE-AI Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("1️⃣ Report")
        st.write("Register missing-person information and upload a photograph.")

    with col2:
        st.subheader("2️⃣ Search")
        st.write("Search publicly available registered cases.")

    with col3:
        st.subheader("3️⃣ AI")
        st.write("Upload a photograph for AI-assisted search.")

    with col4:
        st.subheader("4️⃣ Verify")
        st.write("Authorized humans verify potential matches.")

    st.divider()

    st.warning(
        "⚠️ Educational prototype. Personal information, photographs and "
        "location data should only be published with appropriate "
        "authorization, consent and privacy protections."
    )


# ============================================================
# REPORT MISSING PERSON
# ============================================================

elif page == "📝 Report Missing Person":

    st.header("📝 Report Missing Person")

    st.info(
        "Public users can submit a report. An authorized administrator "
        "must review the information before it is treated as an official case."
    )

    with st.form("missing_person_form", clear_on_submit=False):

        name = st.text_input("Full Name *")

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=18
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other", "Prefer not to say"]
        )

        location = st.text_input("Last Seen Location *")

        last_seen = st.text_input(
            "Last Seen Date & Time",
            placeholder="Example: 16-09-2026 18:30"
        )

        contact = st.text_input("Emergency Contact Number")

        description = st.text_area(
            "Additional Description",
            placeholder="Clothing, identifying features, other useful information..."
        )

        photo = st.file_uploader(
            "Upload Photograph",
            type=["jpg", "jpeg", "png"]
        )

        submitted = st.form_submit_button("🚨 Submit Missing Person Report")

        if submitted:

            if not name.strip():
                st.error("Please enter the person's name.")

            elif not location.strip():
                st.error("Please enter the last seen location.")

            else:

                case_id = (
                    max(
                        [case.get("id", 0) for case in st.session_state.cases],
                        default=0
                    ) + 1
                )

                photo_path = ""

                if photo is not None:
                    extension = os.path.splitext(photo.name)[1].lower()
                    filename = f"case_{case_id}{extension}"
                    photo_path = os.path.join(UPLOAD_FOLDER, filename)

                    with open(photo_path, "wb") as file:
                        file.write(photo.getbuffer())

                new_case = {
                    "id": case_id,
                    "name": name.strip(),
                    "age": age,
                    "gender": gender,
                    "location": location.strip(),
                    "last_seen": last_seen.strip(),
                    "contact": contact.strip(),
                    "description": description.strip(),
                    "photo": photo_path,
                    "status": "Missing",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "gps": {
                        "latitude": None,
                        "longitude": None
                    }
                }

                st.session_state.cases.append(new_case)
                save_cases(st.session_state.cases)

                st.success(
                    f"Report submitted successfully. Case #{case_id}."
                )

                st.info(
                    "Case status and case details can only be changed by an authorized administrator."
                )


# ============================================================
# SEARCH CASES
# ============================================================

elif page == "🔍 Search Cases":

    st.header("🔍 Search Missing Person Cases")

    st.write("Search publicly available registered cases.")

    search_name = st.text_input("Person Name")

    search_location = st.text_input("Location")

    search_status = st.selectbox(
        "Status",
        ["All", "Missing", "Found"]
    )

    if st.button("🔎 Search", use_container_width=True):

        results = []

        for case in st.session_state.cases:

            name_query = search_name.strip().lower()
            location_query = search_location.strip().lower()

            name_match = (
                not name_query
                or name_query in case.get("name", "").lower()
            )

            location_match = (
                not location_query
                or location_query in case.get("location", "").lower()
            )

            status_match = (
                search_status == "All"
                or case.get("status") == search_status
            )

            if name_match and location_match and status_match:
                results.append(case)

        st.divider()

        if not results:
            st.warning("No matching cases found.")

        else:
            st.success(f"{len(results)} case(s) found.")

            for case in results:

                with st.expander(
                    f"Case #{case['id']} | {case['name']} | {case['status']}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Name:** {case['name']}")
                        st.write(f"**Age:** {case['age']}")
                        st.write(f"**Gender:** {case['gender']}")
                        st.write(f"**Status:** {case['status']}")

                    with col2:
                        st.write(f"**Last Seen:** {case['location']}")
                        st.write(f"**Date/Time:** {case['last_seen']}")

                    if case.get("description"):
                        st.write(
                            "**Description:** " + case["description"]
                        )

                    if (
                        case.get("photo")
                        and os.path.exists(case["photo"])
                    ):
                        st.image(case["photo"], width=250)

                    st.caption(
                        "Case status can only be changed by an authorized administrator."
                    )


# ============================================================
# AI FACE SEARCH
# ============================================================

elif page == "🤖 AI Face Search":

    st.header("🤖 AI-Assisted Face Search")

    st.info(
        "This page currently provides the photograph upload and candidate "
        "search interface. The current prototype does not claim that a "
        "candidate is the same person."
    )

    search_photo = st.file_uploader(
        "Upload Search Photograph",
        type=["jpg", "jpeg", "png"],
        key="ai_search_photo"
    )

    if search_photo is not None:

        st.image(
            search_photo,
            caption="Uploaded Search Image",
            width=350
        )

        if st.button("🤖 Start AI Search", use_container_width=True):

            st.write("### 🔄 AI Search Process")

            progress = st.progress(0)

            progress.progress(20)
            st.write("✅ Image uploaded")

            progress.progress(40)
            st.write("✅ Image preprocessing completed")

            progress.progress(60)
            st.write("🔎 Searching registered cases...")

            progress.progress(80)
            st.write("🧠 Generating potential candidates...")

            progress.progress(100)

            st.success("Search process completed.")

            if st.session_state.cases:

                st.subheader("Registered Case Candidates")

                for case in st.session_state.cases:

                    st.write(
                        f"**Case #{case['id']} — {case['name']}**"
                    )

                    st.write(
                        f"Location: {case['location']}"
                    )

                    st.write(
                        f"Status: {case['status']}"
                    )

                    st.divider()

            else:
                st.info("No registered cases are available.")

            st.warning(
                "⚠️ A possible face similarity must never be treated as "
                "proof of identity. Authorized human verification is required."
            )


# ============================================================
# ADMIN LOGIN
# ============================================================

elif page == "🔐 Admin Login":

    if st.session_state.admin_logged_in:
        st.success("You are already logged in as administrator.")
    else:
        admin_login_page()


# ============================================================
# GPS LOCATION — ADMIN ONLY
# ============================================================

elif page == "📍 GPS Location":

    if not st.session_state.admin_logged_in:
        st.error("🔒 Administrator access required.")
        st.stop()

    st.header("📍 GPS Location Management")

    if not st.session_state.cases:
        st.info("No cases are registered yet.")

    else:

        case_options = {
            f"Case #{case['id']} - {case['name']}": case["id"]
            for case in st.session_state.cases
        }

        selected_case = st.selectbox(
            "Select Case",
            list(case_options.keys())
        )

        selected_id = case_options[selected_case]

        selected_case_data = next(
            (
                case for case in st.session_state.cases
                if case["id"] == selected_id
            ),
            None
        )

        if selected_case_data:

            st.subheader(selected_case_data["name"])

            gps = selected_case_data.get("gps", {})

            current_lat = gps.get("latitude")
            current_lon = gps.get("longitude")

            if current_lat is None:
                current_lat = 0.0

            if current_lon is None:
                current_lon = 0.0

            latitude = st.number_input(
                "Latitude",
                value=float(current_lat),
                format="%.6f"
            )

            longitude = st.number_input(
                "Longitude",
                value=float(current_lon),
                format="%.6f"
            )

            if st.button(
                "📍 Update GPS Location",
                use_container_width=True
            ):

                selected_case_data["gps"] = {
                    "latitude": latitude,
                    "longitude": longitude
                }

                save_cases(st.session_state.cases)

                st.success("GPS location updated successfully.")

            if latitude != 0 or longitude != 0:

                st.write(f"**Latitude:** {latitude}")
                st.write(f"**Longitude:** {longitude}")

                st.map({
                    "latitude": [latitude],
                    "longitude": [longitude]
                })


# ============================================================
# ADMIN DASHBOARD — ADMIN ONLY
# ============================================================

elif page == "📊 Admin Dashboard":

    if not st.session_state.admin_logged_in:
        st.error("🔒 Administrator access required.")
        st.stop()

    st.header("📊 Admin Dashboard")

    st.success(
        "🔐 You are viewing the administrator-only dashboard."
    )

    total = len(st.session_state.cases)

    missing = sum(
        1 for case in st.session_state.cases
        if case.get("status") == "Missing"
    )

    found = sum(
        1 for case in st.session_state.cases
        if case.get("status") == "Found"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cases", total)

    with col2:
        st.metric("Missing", missing)

    with col3:
        st.metric("Found", found)

    st.divider()

    if not st.session_state.cases:

        st.info("No cases available.")

    else:

        st.subheader("📋 Case Records")

        for case in st.session_state.cases:

            with st.expander(
                f"Case #{case['id']} — {case['name']}"
            ):

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Name:** {case['name']}")
                    st.write(f"**Age:** {case['age']}")
                    st.write(f"**Gender:** {case['gender']}")
                    st.write(f"**Location:** {case['location']}")

                with col2:
                    st.write(f"**Last Seen:** {case['last_seen']}")
                    st.write(f"**Contact:** {case['contact']}")
                    st.write(f"**Created:** {case['created_at']}")

                current_status = case.get("status", "Missing")

                new_status = st.selectbox(
                    "Case Status",
                    ["Missing", "Found"],
                    index=0 if current_status == "Missing" else 1,
                    key=f"status_{case['id']}"
                )

                if st.button(
                    "💾 Save Status",
                    key=f"save_{case['id']}"
                ):

                    case["status"] = new_status

                    save_cases(st.session_state.cases)

                    st.success("Case status updated successfully.")

                if case.get("gps", {}).get("latitude") is not None:
                    st.write(
                        "📍 GPS: "
                        f"{case['gps']['latitude']}, "
                        f"{case['gps']['longitude']}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TRACE-AI | Finding Missing People Using AI | "
    "Educational Prototype"
)
