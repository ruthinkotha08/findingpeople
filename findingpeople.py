import streamlit as st
import json
import os
import cv2
import numpy as np
from datetime import datetime

# ============================================================
# TRACE-AI - Finding Missing People Using AI
# ============================================================

st.set_page_config(
    page_title="TRACE-AI",
    page_icon="🔎",
    layout="wide"
)

DATA_FILE = "cases.json"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# DATABASE
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
        json.dump(cases, file, indent=4)


if "cases" not in st.session_state:
    st.session_state.cases = load_cases()


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def read_image(file):

    data = np.frombuffer(
        file,
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Select largest face
    face = max(
        faces,
        key=lambda x: x[2] * x[3]
    )

    x, y, w, h = face

    crop = gray[
        y:y+h,
        x:x+w
    ]

    crop = cv2.resize(
        crop,
        (200, 200)
    )

    crop = cv2.equalizeHist(crop)

    return crop


# ============================================================
# CREATE FACE RECOGNITION MODEL
# ============================================================

def create_model():

    if not hasattr(cv2, "face"):

        return None, (
            "OpenCV face recognition module is not installed. "
            "Check streamlit opencv-contrib-python-headless numpy."
        )

    images = []
    labels = []

    valid_cases = []

    for case in st.session_state.cases:

        photo = case.get("photo", "")

        if not photo:
            continue

        if not os.path.exists(photo):
            continue

        image = cv2.imread(photo)

        if image is None:
            continue

        face = detect_face(image)

        if face is None:
            continue

        images.append(face)

        labels.append(
            int(case["id"])
        )

        valid_cases.append(case)

    if not images:

        return None, (
            "No registered photographs with detectable faces."
        )

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(
        images,
        np.array(
            labels,
            dtype=np.int32
        )
    )

    return model, valid_cases


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(distance):

    # LBPH distance:
    # lower = more similar

    score = 100 - (
        distance * 0.75
    )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return score


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 TRACE-AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📝 Report Missing Person",
        "🔍 Search Cases",
        "🤖 AI Face Search",
        "📍 GPS Location",
        "🚨 Alerts",
        "📊 Admin Dashboard"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🔎 TRACE-AI")

    st.subheader(
        "Finding Missing People Using AI"
    )

    st.write(
        "An AI-assisted system for reporting, searching "
        "and locating missing people."
    )

    st.divider()

    total = len(
        st.session_state.cases
    )

    missing = sum(
        c.get("status") == "Missing"
        for c in st.session_state.cases
    )

    found = sum(
        c.get("status") == "Found"
        for c in st.session_state.cases
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Cases",
        total
    )

    col2.metric(
        "Missing",
        missing
    )

    col3.metric(
        "Found",
        found
    )

    st.divider()

    st.header("How TRACE-AI Works")

    a, b, c, d = st.columns(4)

    a.subheader("1️⃣ Report")
    a.write(
        "Register the missing person and photograph."
    )

    b.subheader("2️⃣ Search")
    b.write(
        "Search registered cases."
    )

    c.subheader("3️⃣ AI")
    c.write(
        "Compare uploaded faces with registered photographs."
    )

    d.subheader("4️⃣ GPS")
    d.write(
        "Store authorized GPS coordinates."
    )

    st.warning(
        "AI matching provides potential leads only. "
        "Human verification is required."
    )


# ============================================================
# REPORT
# ============================================================

elif page == "📝 Report Missing Person":

    st.header(
        "📝 Report Missing Person"
    )

    with st.form("report_form"):

        name = st.text_input(
            "Full Name *"
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=18
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other",
                "Prefer not to say"
            ]
        )

        location = st.text_input(
            "Last Seen Location *"
        )

        last_seen = st.text_input(
            "Last Seen Date & Time"
        )

        contact = st.text_input(
            "Contact Number"
        )

        description = st.text_area(
            "Description"
        )

        photo = st.file_uploader(
            "Upload Clear Face Photograph *",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        submit = st.form_submit_button(
            "🚨 Register Missing Person"
        )

    if submit:

        if not name.strip():

            st.error(
                "Enter the person's name."
            )

        elif not location.strip():

            st.error(
                "Enter the last seen location."
            )

        elif photo is None:

            st.error(
                "Upload a photograph."
            )

        else:

            image = read_image(
                photo.getvalue()
            )

            if image is None:

                st.error(
                    "Invalid image."
                )

            else:

                face = detect_face(
                    image
                )

                if face is None:

                    st.error(
                        "No face detected. "
                        "Please upload a clear front-facing photograph."
                    )

                else:

                    case_id = max(
                        [
                            c.get("id", 0)
                            for c in st.session_state.cases
                        ],
                        default=0
                    ) + 1

                    extension = os.path.splitext(
                        photo.name
                    )[1].lower()

                    filename = (
                        f"case_{case_id}"
                        f"{extension}"
                    )

                    path = os.path.join(
                        UPLOAD_DIR,
                        filename
                    )

                    with open(
                        path,
                        "wb"
                    ) as file:

                        file.write(
                            photo.getbuffer()
                        )

                    case = {

                        "id": case_id,

                        "name": name,

                        "age": age,

                        "gender": gender,

                        "location": location,

                        "last_seen": last_seen,

                        "contact": contact,

                        "description": description,

                        "photo": path,

                        "status": "Missing",

                        "created_at":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "gps": {
                            "latitude": None,
                            "longitude": None
                        }
                    }

                    st.session_state.cases.append(
                        case
                    )

                    save_cases(
                        st.session_state.cases
                    )

                    st.success(
                        f"Case #{case_id} registered successfully!"
                    )

                    st.image(
                        image,
                        channels="BGR",
                        width=250
                    )


# ============================================================
# SEARCH
# ============================================================

elif page == "🔍 Search Cases":

    st.header(
        "🔍 Search Cases"
    )

    name_search = st.text_input(
        "Search Name"
    )

    location_search = st.text_input(
        "Search Location"
    )

    status_search = st.selectbox(
        "Status",
        [
            "All",
            "Missing",
            "Found"
        ]
    )

    if st.button(
        "🔎 Search",
        use_container_width=True
    ):

        results = []

        for case in st.session_state.cases:

            name_match = (
                not name_search
                or
                name_search.lower()
                in case["name"].lower()
            )

            location_match = (
                not location_search
                or
                location_search.lower()
                in case["location"].lower()
            )

            status_match = (
                status_search == "All"
                or
                status_search == case["status"]
            )

            if (
                name_match
                and
                location_match
                and
                status_match
            ):

                results.append(
                    case
                )

        if not results:

            st.warning(
                "No matching cases found."
            )

        else:

            st.success(
                f"{len(results)} case(s) found."
            )

            for case in results:

                with st.expander(
                    f"Case #{case['id']} — "
                    f"{case['name']}"
                ):

                    st.write(
                        f"**Age:** {case['age']}"
                    )

                    st.write(
                        f"**Location:** {case['location']}"
                    )

                    st.write(
                        f"**Status:** {case['status']}"
                    )

                    if case.get("photo") and os.path.exists(
                        case["photo"]
                    ):

                        st.image(
                            case["photo"],
                            width=250
                        )


# ============================================================
# AI FACE SEARCH
# ============================================================

elif page == "🤖 AI Face Search":

    st.header(
        "🤖 AI Face Search"
    )

    st.write(
        "Upload a photograph and compare its detected face "
        "with registered case photographs."
    )

    search_photo = st.file_uploader(
        "Upload Search Photograph",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="search_photo"
    )

    if search_photo:

        image = read_image(
            search_photo.getvalue()
        )

        if image is None:

            st.error(
                "Could not read image."
            )

        else:

            st.image(
                image,
                channels="BGR",
                caption="Search Photograph",
                width=350
            )

            face = detect_face(
                image
            )

            if face is None:

                st.error(
                    "❌ No face detected."
                )

                st.info(
                    "Use a clear photograph where the person's "
                    "face is visible."
                )

            else:

                st.success(
                    "✅ Face detected successfully."
                )

                if st.button(
                    "🔎 FIND SIMILAR FACE",
                    use_container_width=True
                ):

                    with st.spinner(
                        "AI is comparing faces..."
                    ):

                        model, cases = create_model()

                        if model is None:

                            st.error(
                                str(cases)
                            )

                        else:

                            predicted_id, distance = (
                                model.predict(face)
                            )

                            similarity = (
                                calculate_similarity(
                                    distance
                                )
                            )

                            matched_case = None

                            for case in cases:

                                if int(
                                    case["id"]
                                ) == int(
                                    predicted_id
                                ):

                                    matched_case = case
                                    break

                            st.divider()

                            if matched_case is not None:

                                st.subheader(
                                    "🎯 Potential Match Found"
                                )

                                col1, col2 = st.columns(2)

                                with col1:

                                    if os.path.exists(
                                        matched_case["photo"]
                                    ):

                                        st.image(
                                            matched_case["photo"],
                                            width=300
                                        )

                                with col2:

                                    st.write(
                                        f"### 👤 {matched_case['name']}"
                                    )

                                    st.write(
                                        f"**Case ID:** "
                                        f"{matched_case['id']}"
                                    )

                                    st.write(
                                        f"**Age:** "
                                        f"{matched_case['age']}"
                                    )

                                    st.write(
                                        f"**Location:** "
                                        f"{matched_case['location']}"
                                    )

                                    st.write(
                                        f"**Status:** "
                                        f"{matched_case['status']}"
                                    )

                                    st.metric(
                                        "Similarity Score",
                                        f"{similarity:.1f}%"
                                    )

                                if similarity >= 60:

                                    st.success(
                                        "🟢 Potentially similar face detected."
                                    )

                                elif similarity >= 40:

                                    st.warning(
                                        "🟡 Weak potential similarity."
                                    )

                                else:

                                    st.error(
                                        "🔴 Low similarity."
                                    )

                                st.warning(
                                    "⚠️ This is an AI-assisted similarity "
                                    "result, not proof of identity. "
                                    "An authorized human must verify any result."
                                )

                            else:

                                st.warning(
                                    "No registered face matched."
                                )


# ============================================================
# GPS
# ============================================================

elif page == "📍 GPS Location":

    st.header(
        "📍 GPS Location"
    )

    if not st.session_state.cases:

        st.info(
            "No cases available."
        )

    else:

        options = {
            f"Case #{c['id']} - {c['name']}":
            c["id"]
            for c in st.session_state.cases
        }

        selected = st.selectbox(
            "Select Case",
            list(options.keys())
        )

        case_id = options[
            selected
        ]

        case = next(
            c for c in st.session_state.cases
            if c["id"] == case_id
        )

        lat = st.number_input(
            "Latitude",
            value=float(
                case["gps"].get("latitude") or 0
            ),
            format="%.6f"
        )

        lon = st.number_input(
            "Longitude",
            value=float(
                case["gps"].get("longitude") or 0
            ),
            format="%.6f"
        )

        if st.button(
            "📍 Update GPS"
        ):

            case["gps"] = {
                "latitude": lat,
                "longitude": lon
            }

            save_cases(
                st.session_state.cases
            )

            st.success(
                "GPS location updated."
            )

        if lat != 0 or lon != 0:

            st.map(
                {
                    "latitude": [lat],
                    "longitude": [lon]
                }
            )


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.header(
        "🚨 Emergency Alerts"
    )

    active = [
        c for c in st.session_state.cases
        if c.get("status") == "Missing"
    ]

    if not active:

        st.success(
            "No active missing-person cases."
        )

    for case in active:

        st.error(
            f"🚨 Case #{case['id']} — "
            f"{case['name']} — "
            f"{case['location']}"
        )

        if st.button(
            f"📢 Generate SOS — Case #{case['id']}",
            key=f"sos_{case['id']}"
        ):

            st.error(
                f"SOS alert generated for "
                f"Case #{case['id']}."
            )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

elif page == "📊 Admin Dashboard":

    st.header(
        "📊 Admin Dashboard"
    )

    total = len(
        st.session_state.cases
    )

    missing = sum(
        c.get("status") == "Missing"
        for c in st.session_state.cases
    )

    found = sum(
        c.get("status") == "Found"
        for c in st.session_state.cases
    )

    a, b, c = st.columns(3)

    a.metric(
        "Total Cases",
        total
    )

    b.metric(
        "Missing",
        missing
    )

    c.metric(
        "Found",
        found
    )

    st.divider()

    for case in st.session_state.cases:

        with st.expander(
            f"Case #{case['id']} — "
            f"{case['name']}"
        ):

            st.write(
                f"**Location:** {case['location']}"
            )

            st.write(
                f"**Contact:** {case['contact']}"
            )

            status = st.selectbox(
                "Case Status",
                [
                    "Missing",
                    "Found"
                ],
                index=(
                    0
                    if case.get("status") == "Missing"
                    else 1
                ),
                key=f"status_{case['id']}"
            )

            if st.button(
                "💾 Save Status",
                key=f"save_{case['id']}"
            ):

                case["status"] = status

                save_cases(
                    st.session_state.cases
                )

                st.success(
                    "Status updated."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TRACE-AI | Finding Missing People Using AI | "
    "Educational Prototype"
)
