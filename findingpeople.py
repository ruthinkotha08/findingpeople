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

# Demo admin credentials
ADMIN_USER = "admin"
ADMIN_PASSWORD = "traceai123"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "cases" not in st.session_state:
    st.session_state.cases = []


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


def save_cases():

    with open(DATA_FILE, "w", encoding="utf-8") as file:

        json.dump(
            st.session_state.cases,
            file,
            indent=4
        )


if not st.session_state.cases:

    st.session_state.cases = load_cases()


# ============================================================
# FACE DETECTOR
# ============================================================

# ============================================================
# FACE DETECTOR
# ============================================================

CASCADE_FILE = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

FACE_CASCADE = cv2.CascadeClassifier(CASCADE_FILE)

if FACE_CASCADE.empty():
    st.error(
        "❌ ERROR: Haar Cascade face detector could not be loaded."
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image):

    if image is None:
        return None

    if FACE_CASCADE.empty():
        return None

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # Select largest face
        x, y, w, h = max(
            faces,
            key=lambda p: p[2] * p[3]
        )

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (200, 200)
        )

        return face

    except cv2.error as e:

        st.error(
            "❌ OpenCV face detection failed."
        )

        st.code(str(e))

        return None

# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def read_image(data):

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )


def detect_face(image):

    if image is None:
        return None

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

    x, y, w, h = max(
        faces,
        key=lambda p: p[2] * p[3]
    )

    face = gray[
        y:y + h,
        x:x + w
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    face = cv2.equalizeHist(face)

    return face


# ============================================================
# CREATE FACE RECOGNITION MODEL
# ============================================================

def create_model():

    if not hasattr(cv2, "face"):

        return (
            None,
            "OpenCV face module is missing. "
            "Install opencv-contrib-python-headless."
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

        return (
            None,
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
# SIMILARITY SCORE
# ============================================================

def calculate_similarity(distance):

    # Lower LBPH distance = more similar

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
# ADMIN LOGIN
# ============================================================

def admin_login():

    st.header("🔐 Admin Login")

    st.info(
        "Administrator access is required "
        "for case management."
    )

    with st.form("admin_login_form"):

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.form_submit_button(
            "🔑 Login",
            use_container_width=True
        )

    if login:

        if (
            username == ADMIN_USER
            and
            password == ADMIN_PASSWORD
        ):

            st.session_state.admin_logged_in = True

            st.success(
                "Login successful."
            )

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 TRACE-AI")

pages = [

    "🏠 Home",

    "🔍 Search Cases",

    "🤖 AI Face Search",

    "📍 GPS Location",

    "🚨 Alerts"

]

# Admin pages appear only after login

if st.session_state.admin_logged_in:

    pages += [

        "📝 Report Missing Person",

        "📊 Admin Dashboard"

    ]

page = st.sidebar.radio(
    "Navigation",
    pages
)


# ============================================================
# ADMIN LOGIN BUTTON
# ============================================================

if st.session_state.admin_logged_in:

    st.sidebar.success(
        "👤 Admin logged in"
    )

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.admin_logged_in = False

        st.rerun()

else:

    if st.sidebar.button(
        "🔐 Admin Login",
        use_container_width=True
    ):

        st.session_state.show_login = True


# ============================================================
# LOGIN SCREEN
# ============================================================

if (
    st.session_state.get(
        "show_login",
        False
    )
    and
    not st.session_state.admin_logged_in
):

    admin_login()

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🔎 TRACE-AI"
    )

    st.subheader(
        "Finding Missing People Using AI"
    )

    st.write(
        "An AI-assisted system for reporting, "
        "searching and locating missing people."
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

    st.header(
        "How TRACE-AI Works"
    )

    a, b, c, d = st.columns(4)

    a.subheader(
        "1️⃣ Report"
    )

    a.write(
        "Admin registers the missing person "
        "and photograph."
    )

    b.subheader(
        "2️⃣ Search"
    )

    b.write(
        "Search cases using name or "
        "location name."
    )

    c.subheader(
        "3️⃣ AI"
    )

    c.write(
        "Compare an uploaded face with "
        "registered photographs."
    )

    d.subheader(
        "4️⃣ GPS"
    )

    d.write(
        "Authorized GPS coordinates can "
        "be stored for a case."
    )

    st.warning(
        "⚠️ AI matching provides potential leads only. "
        "Human verification is required."
    )


# ============================================================
# SEARCH CASES
# ============================================================

elif page == "🔍 Search Cases":

    st.header(
        "🔍 Search Missing-Person Cases"
    )

    st.write(
        "Search using a person's name or "
        "location name."
    )

    st.info(
        "Example location: Hyderabad, "
        "Secunderabad, Vijayawada"
    )

    name_search = st.text_input(
        "👤 Person Name",
        placeholder="Example: Raaga"
    )

    location_search = st.text_input(
        "📍 Location Name",
        placeholder="Example: Hyderabad"
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

                not name_search.strip()

                or

                name_search.lower()
                in
                case.get(
                    "name",
                    ""
                ).lower()

            )

            location_match = (

                not location_search.strip()

                or

                location_search.lower()
                in
                case.get(
                    "location",
                    ""
                ).lower()

            )

            status_match = (

                status_search == "All"

                or

                status_search
                ==
                case.get(
                    "status"
                )

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

                with st.container(
                    border=True
                ):

                    col1, col2 = st.columns(
                        [1, 2]
                    )

                    with col1:

                        if (
                            case.get("photo")
                            and
                            os.path.exists(
                                case["photo"]
                            )
                        ):

                            st.image(
                                case["photo"],
                                width=220
                            )

                    with col2:

                        st.subheader(
                            f"Case #{case['id']} — "
                            f"{case['name']}"
                        )

                        st.write(
                            f"**Age:** "
                            f"{case.get('age', '')}"
                        )

                        st.write(
                            f"**Gender:** "
                            f"{case.get('gender', '')}"
                        )

                        st.write(
                            f"**Last Seen Location:** "
                            f"{case.get('location', '')}"
                        )

                        st.write(
                            f"**Last Seen:** "
                            f"{case.get('last_seen', '')}"
                        )

                        st.write(
                            f"**Status:** "
                            f"{case.get('status', '')}"
                        )

                        if case.get(
                            "description"
                        ):

                            st.write(
                                f"**Description:** "
                                f"{case['description']}"
                            )


# ============================================================
# REPORT MISSING PERSON
# ============================================================

elif page == "📝 Report Missing Person":

    if not st.session_state.admin_logged_in:

        admin_login()

        st.stop()

    st.header(
        "📝 Report Missing Person"
    )

    with st.form(
        "report_form"
    ):

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
            "Last Seen Location *",
            placeholder="Example: Hyderabad"
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
            "🚨 Register Missing Person",
            use_container_width=True
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
                        "Please upload a clear "
                        "front-facing photograph."
                    )

                else:

                    case_id = max(
                        [
                            int(
                                c.get(
                                    "id",
                                    0
                                )
                            )
                            for c in
                            st.session_state.cases
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

                        "name": name.strip(),

                        "age": int(age),

                        "gender": gender,

                        "location": location.strip(),

                        "last_seen": last_seen.strip(),

                        "contact": contact.strip(),

                        "description":
                            description.strip(),

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

                    save_cases()

                    st.success(
                        f"Case #{case_id} "
                        "registered successfully!"
                    )

                    st.image(
                        image,
                        channels="BGR",
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
        "Upload a photograph to find "
        "potential similarities among "
        "registered cases."
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
                    "Use a clear photograph "
                    "where the person's face "
                    "is visible."
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
                                model.predict(
                                    face
                                )
                            )

                            score = (
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

                            if matched_case:

                                st.divider()

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
                                        f"### 👤 "
                                        f"{matched_case['name']}"
                                    )

                                    st.write(
                                        f"**Case ID:** "
                                        f"{matched_case['id']}"
                                    )

                                    st.write(
                                        f"**Age:** "
                                        f"{matched_case.get('age', '')}"
                                    )

                                    st.write(
                                        f"**Location:** "
                                        f"{matched_case.get('location', '')}"
                                    )

                                    st.write(
                                        f"**Status:** "
                                        f"{matched_case.get('status', '')}"
                                    )

                                    st.metric(
                                        "Similarity Score",
                                        f"{score:.1f}%"
                                    )

                                if score >= 60:

                                    st.success(
                                        "🟢 Potentially similar "
                                        "face detected."
                                    )

                                elif score >= 40:

                                    st.warning(
                                        "🟡 Weak potential similarity."
                                    )

                                else:

                                    st.error(
                                        "🔴 Low similarity."
                                    )

                                st.warning(
                                    "⚠️ This is an AI-assisted "
                                    "similarity result, not proof "
                                    "of identity. Authorized human "
                                    "verification is required."
                                )

                            else:

                                st.warning(
                                    "No registered face matched."
                                )


# ============================================================
# GPS LOCATION
# ============================================================

elif page == "📍 GPS Location":

    st.header(
        "📍 GPS Location"
    )

    st.write(
        "GPS coordinates are used internally "
        "for authorized device/location data."
    )

    st.info(
        "Normal users search using location names. "
        "They do NOT need to enter latitude or longitude."
    )

    if not st.session_state.admin_logged_in:

        st.warning(
            "🔐 Admin login is required "
            "to update GPS data."
        )

    elif not st.session_state.cases:

        st.info(
            "No cases available."
        )

    else:

        options = {

            f"Case #{c['id']} - {c['name']}":
                c["id"]

            for c in
            st.session_state.cases

        }

        selected = st.selectbox(
            "Select Case",
            list(options.keys())
        )

        case_id = options[
            selected
        ]

        case = next(
            c for c in
            st.session_state.cases
            if int(c["id"]) == int(case_id)
        )

        gps = case.setdefault(
            "gps",
            {}
        )

        lat = st.number_input(
            "Latitude",
            value=float(
                gps.get(
                    "latitude"
                )
                or 0
            ),
            format="%.6f"
        )

        lon = st.number_input(
            "Longitude",
            value=float(
                gps.get(
                    "longitude"
                )
                or 0
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

            save_cases()

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

        c for c in
        st.session_state.cases

        if c.get(
            "status"
        ) == "Missing"

    ]

    if not active:

        st.success(
            "No active missing-person cases."
        )

    else:

        for case in active:

            st.error(
                f"🚨 Case #{case['id']} — "
                f"{case['name']} — "
                f"{case.get('location', '')}"
            )

            if st.session_state.admin_logged_in:

                if st.button(
                    f"📢 Generate SOS — "
                    f"Case #{case['id']}",
                    key=f"sos_{case['id']}"
                ):

                    st.error(
                        f"SOS alert generated "
                        f"for Case #{case['id']}."
                    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

elif page == "📊 Admin Dashboard":

    if not st.session_state.admin_logged_in:

        admin_login()

        st.stop()

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

            if (
                case.get("photo")
                and
                os.path.exists(
                    case["photo"]
                )
            ):

                st.image(
                    case["photo"],
                    width=220
                )

            st.write(
                f"**Location:** "
                f"{case.get('location', '')}"
            )

            st.write(
                f"**Contact:** "
                f"{case.get('contact', '')}"
            )

            st.write(
                f"**Description:** "
                f"{case.get('description', '')}"
            )

            status = st.selectbox(
                "Case Status",
                [
                    "Missing",
                    "Found"
                ],
                index=(
                    0
                    if case.get(
                        "status"
                    ) == "Missing"
                    else 1
                ),
                key=f"status_{case['id']}"
            )

            if st.button(
                "💾 Save Status",
                key=f"save_{case['id']}"
            ):

                case["status"] = status

                save_cases()

                st.success(
                    "Status updated."
                )

            if st.button(
                "🗑️ Delete Case",
                key=f"delete_{case['id']}"
            ):

                photo_path = case.get(
                    "photo",
                    ""
                )

                st.session_state.cases = [

                    x for x in
                    st.session_state.cases

                    if int(x["id"])
                    !=
                    int(case["id"])

                ]

                if (
                    photo_path
                    and
                    os.path.exists(
                        photo_path
                    )
                ):

                    os.remove(
                        photo_path
                    )

                save_cases()

                st.success(
                    "Case deleted."
                )

                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TRACE-AI | Finding Missing People Using AI | "
    "Educational Prototype"
)
