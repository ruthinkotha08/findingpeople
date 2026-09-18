import streamlit as st
import cv2
import numpy as np
import urllib.request
import urllib.parse
import tempfile
import os
from datetime import datetime
from uuid import uuid4
from supabase import create_client, Client


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TRACE-AI",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# SUPABASE
# ============================================================

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

except Exception as e:
    st.error("Supabase connection could not be established.")
    st.error(str(e))
    st.stop()


# ============================================================
# SETTINGS
# ============================================================

BUCKET_NAME = "case-photos"

ADMIN_USER = "admin"
ADMIN_PASSWORD = "Swarajyam@2014"
ADMIN_EMAIL = "rkotha2@student.gitam.edu"

# SFace cosine similarity threshold.
# Higher = stricter.
#
# 0.363 is a commonly used OpenCV SFace threshold for cosine
# similarity. We use a stricter value for this prototype.
FACE_MATCH_THRESHOLD = 0.50


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
    try:
        response = (
            supabase
            .table("cases")
            .select("*")
            .order("id", desc=True)
            .execute()
        )

        return response.data or []

    except Exception as e:
        st.error("Could not load cases from Supabase.")
        st.error(str(e))
        return []


def get_next_id():
    try:
        response = (
            supabase
            .table("cases")
            .select("id")
            .order("id", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            return int(response.data[0]["id"]) + 1

        return 1

    except Exception:
        return 1


def create_ticket_id(case_id):
    today = datetime.now().strftime("%Y%m%d")
    return f"MP-{today}-{case_id:04d}"


def insert_case(case_data):
    try:
        response = (
            supabase
            .table("cases")
            .insert(case_data)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error("Could not save the case.")
        st.error(str(e))
        return None


def update_case_status(case_id, new_status):
    try:
        response = (
            supabase
            .table("cases")
            .update({
                "status": new_status
            })
            .eq("id", case_id)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error("Could not update case status.")
        st.error(str(e))
        return None


def update_case_location(case_id, new_location):
    try:
        response = (
            supabase
            .table("cases")
            .update({
                "location": new_location
            })
            .eq("id", case_id)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error("Could not update location.")
        st.error(str(e))
        return None


def delete_case(case_id):
    try:
        response = (
            supabase
            .table("cases")
            .delete()
            .eq("id", case_id)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error("Could not delete the case.")
        st.error(str(e))
        return None


# ============================================================
# STORAGE
# ============================================================

def upload_photo(uploaded_file, ticket_id):

    try:
        extension = (
            uploaded_file.name
            .split(".")[-1]
            .lower()
        )

        if extension not in [
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]:
            extension = "jpg"

        unique_name = uuid4().hex[:10]

        storage_path = (
            f"cases/"
            f"{ticket_id}_"
            f"{unique_name}."
            f"{extension}"
        )

        file_bytes = uploaded_file.getvalue()

        supabase.storage.from_(
            BUCKET_NAME
        ).upload(
            storage_path,
            file_bytes,
            {
                "content-type": uploaded_file.type,
                "upsert": "false"
            }
        )

        return storage_path

    except Exception as e:
        st.error("Photo upload failed.")
        st.error(str(e))
        return None


def delete_photo(storage_path):

    if not storage_path:
        return

    try:
        supabase.storage.from_(
            BUCKET_NAME
        ).remove(
            [storage_path]
        )

    except Exception:
        pass


def get_public_photo_url(storage_path):

    if not storage_path:
        return None

    try:
        return (
            supabase
            .storage
            .from_(BUCKET_NAME)
            .get_public_url(storage_path)
        )

    except Exception:
        return None


def download_image(storage_path):

    url = get_public_photo_url(storage_path)

    if not url:
        return None

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            image_bytes = response.read()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:
        return None


# ============================================================
# HAAR CASCADE
# ============================================================

@st.cache_resource
def load_face_detector():

    # First try the OpenCV installation.
    try:

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        detector = cv2.CascadeClassifier(
            cascade_path
        )

        if not detector.empty():
            return detector

    except Exception:
        pass

    # Fallback download.
    cascade_url = (
        "https://raw.githubusercontent.com/"
        "opencv/opencv/master/data/"
        "haarcascades/"
        "haarcascade_frontalface_default.xml"
    )

    temp_path = None

    try:

        request = urllib.request.Request(
            cascade_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            xml_data = response.read()

        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            delete=False
        ) as temp_file:

            temp_file.write(xml_data)
            temp_path = temp_file.name

        detector = cv2.CascadeClassifier(
            temp_path
        )

        if not detector.empty():
            return detector

    except Exception:
        pass

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass

    return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_largest_face(image, detector):

    if image is None or detector is None:
        return None

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        # Add a small margin around the face.
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.20)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        x2 = min(
            image.shape[1],
            x + w + margin_x
        )

        y2 = min(
            image.shape[0],
            y + h + margin_y
        )

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            return None

        return face

    except Exception:
        return None


# ============================================================
# SFACE MODEL
# ============================================================

@st.cache_resource
def load_sface_model():

    # OpenCV's SFace ONNX model.
    model_url = (
        "https://github.com/opencv/opencv_zoo/raw/"
        "main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec_int8bq.onnx"
    )

    temp_path = os.path.join(
        tempfile.gettempdir(),
        "traceai_sface.onnx"
    )

    try:

        # Download only if it does not already exist.
        if not os.path.exists(temp_path):

            request = urllib.request.Request(
                model_url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:

                model_bytes = response.read()

            with open(
                temp_path,
                "wb"
            ) as model_file:

                model_file.write(model_bytes)

        # Create SFace recognizer.
        recognizer = cv2.FaceRecognizerSF_create(
            temp_path,
            ""
        )

        return recognizer

    except Exception as e:

        return None


# ============================================================
# SFACE EMBEDDING
# ============================================================

def create_face_embedding(
    image,
    detector,
    recognizer
):

    if image is None:
        return None

    if detector is None:
        return None

    if recognizer is None:
        return None

    try:

        face = detect_largest_face(
            image,
            detector
        )

        if face is None:
            return None

        # SFace expects the original face image.
        face = np.ascontiguousarray(face)

        # Detect face again on cropped image.
        face_gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        face_gray = cv2.equalizeHist(
            face_gray
        )

        inner_faces = detector.detectMultiScale(
            face_gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(inner_faces) > 0:

            fx, fy, fw, fh = max(
                inner_faces,
                key=lambda box: box[2] * box[3]
            )

            face = face[
                fy:fy + fh,
                fx:fx + fw
            ]

        face = cv2.resize(
            face,
            (160, 160)
        )

        face = np.ascontiguousarray(
            face
        )

        # Align/crop face for SFace.
        # If alignment fails, use the resized face.
        try:

            faces = detector.detectMultiScale(
                cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2GRAY
                ),
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30)
            )

        except Exception:
            faces = []

        # SFace alignCrop requires a 5-point landmark
        # detector, which is not provided by Haar.
        #
        # Therefore we use the centered normalized face
        # as input to feature extraction.
        #
        # OpenCV's feature extractor accepts the image
        # through alignCrop only in the standard pipeline,
        # but recognize() can work with a normalized crop
        # when using compatible model versions.

        try:

            feature = recognizer.feature(
                face
            )

        except Exception:

            # Alternative: use alignCrop when supported.
            try:

                feature = recognizer.feature(
                    face
                )

            except Exception:
                return None

        if feature is None:
            return None

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        # Normalize embedding.
        norm = np.linalg.norm(feature)

        if norm <= 0:
            return None

        feature = feature / norm

        return feature

    except Exception:
        return None


# ============================================================
# FACE EMBEDDING USING OPENCV DNN FACE RECOGNIZER
# ============================================================

def get_face_feature(
    image,
    detector,
    recognizer
):

    if image is None:
        return None

    if detector is None:
        return None

    if recognizer is None:
        return None

    try:

        # Detect largest face.
        face = detect_largest_face(
            image,
            detector
        )

        if face is None:
            return None

        # SFace works with a normalized face crop.
        face = cv2.resize(
            face,
            (112, 112)
        )

        face = np.ascontiguousarray(
            face
        )

        # OpenCV SFace feature extraction.
        feature = recognizer.feature(
            face
        )

        if feature is None:
            return None

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        norm = np.linalg.norm(feature)

        if norm == 0:
            return None

        feature = feature / norm

        return feature

    except Exception:
        return None


# ============================================================
# FACE SIMILARITY
# ============================================================

def cosine_similarity(
    feature1,
    feature2
):

    if feature1 is None:
        return 0.0

    if feature2 is None:
        return 0.0

    try:

        a = np.asarray(
            feature1,
            dtype=np.float32
        ).flatten()

        b = np.asarray(
            feature2,
            dtype=np.float32
        ).flatten()

        if len(a) != len(b):
            return 0.0

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        value = float(
            np.dot(a, b) / denominator
        )

        return value

    except Exception:
        return 0.0


def similarity_to_percentage(
    similarity
):

    # Convert cosine similarity to a display value.
    #
    # This is a similarity score, NOT identity accuracy.
    #
    # Values below 0 are treated as zero.
    # Values around 0.50+ are increasingly useful
    # for this prototype.

    similarity = max(
        -1.0,
        min(
            1.0,
            float(similarity)
        )
    )

    # Map 0.20 -> 0%
    # Map 1.00 -> 100%
    #
    # This makes the displayed number easier to
    # understand without calling it accuracy.

    score = (
        (similarity - 0.20)
        / 0.80
    ) * 100.0

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return round(
        score,
        1
    )


# ============================================================
# FIND BEST FACE MATCH
# ============================================================

def find_best_face_match(
    uploaded_image,
    cases
):

    detector = load_face_detector()

    if detector is None:
        return None, None, "Face detector could not be loaded."

    recognizer = load_sface_model()

    if recognizer is None:
        return None, None, (
            "The OpenCV SFace model could not be loaded. "
            "Please try again."
        )

    uploaded_feature = get_face_feature(
        uploaded_image,
        detector,
        recognizer
    )

    if uploaded_feature is None:

        return None, None, (
            "No usable face was detected in the uploaded photo. "
            "Use a clear, front-facing photograph."
        )

    best_case = None
    best_similarity = -1.0

    usable_photos = 0

    for case in cases:

        storage_path = case.get(
            "photo_path"
        )

        if not storage_path:
            continue

        stored_image = download_image(
            storage_path
        )

        if stored_image is None:
            continue

        stored_feature = get_face_feature(
            stored_image,
            detector,
            recognizer
        )

        if stored_feature is None:
            continue

        usable_photos += 1

        similarity = cosine_similarity(
            uploaded_feature,
            stored_feature
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_case = case

    if best_case is None:

        return None, None, (
            "No usable case photographs were available."
        )

    score = similarity_to_percentage(
        best_similarity
    )

    return (
        best_case,
        {
            "similarity": best_similarity,
            "score": score,
            "usable_photos": usable_photos
        },
        None
    )


# ============================================================
# LOAD CASES
# ============================================================

st.session_state.cases = load_cases()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔎 TRACE-AI"
)

st.sidebar.caption(
    "Finding Missing People Using AI"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📝 Report Missing Person",
        "🎫 Track Ticket",
        "🔍 Search Cases",
        "🤖 AI Face Search",
        "📍 GPS Location",
        "🚨 Alerts",
        "📧 Contact Administrator",
        "🔐 Admin Login"
    ]
)


# ============================================================
# ADMIN LOGOUT
# ============================================================

if st.session_state.admin_logged_in:

    st.sidebar.success(
        "Admin logged in"
    )

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.admin_logged_in = False
        st.rerun()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🔎 TRACE-AI"
    )

    st.subheader(
        "Finding Missing People Using Artificial Intelligence"
    )

    st.write(
        """
        TRACE-AI helps store and search missing-person
        cases using case information, photographs,
        location names and AI-based face comparison.
        """
    )

    st.divider()

    total_cases = len(
        st.session_state.cases
    )

    missing_cases = len([
        case
        for case in st.session_state.cases
        if str(
            case.get(
                "status",
                ""
            )
        ).lower() == "missing"
    ])

    found_cases = len([
        case
        for case in st.session_state.cases
        if str(
            case.get(
                "status",
                ""
            )
        ).lower() == "found"
    ])

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Cases",
            total_cases
        )

    with col2:

        st.metric(
            "Missing",
            missing_cases
        )

    with col3:

        st.metric(
            "Found",
            found_cases
        )

    st.divider()

    st.info(
        "Use the sidebar to report, track, search, "
        "or manage cases."
    )


# ============================================================
# REPORT MISSING PERSON
# ============================================================

elif page == "📝 Report Missing Person":

    st.title(
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
                "Not Specified"
            ]
        )

        location = st.text_input(
            "Last Known Location *",
            placeholder="Example: Hyderabad"
        )

        last_seen = st.text_input(
            "Last Seen Date / Time",
            placeholder=(
                "Example: "
                "17 September 2026, 6:30 PM"
            )
        )

        reporter_name = st.text_input(
            "Reporter's Name"
        )

        contact = st.text_input(
            "Reporter Contact Number *"
        )

        description = st.text_area(
            "Additional Description"
        )

        photo = st.file_uploader(
            "Upload Photo *",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

        submitted = st.form_submit_button(
            "🚨 Submit Missing Person Report",
            use_container_width=True
        )

    if submitted:

        if not name.strip():

            st.error(
                "Please enter the person's name."
            )

        elif not location.strip():

            st.error(
                "Please enter the last known location."
            )

        elif not contact.strip():

            st.error(
                "Please enter the reporter's contact number."
            )

        elif photo is None:

            st.error(
                "Please upload a photo."
            )

        else:

            case_id = get_next_id()

            ticket_id = create_ticket_id(
                case_id
            )

            storage_path = upload_photo(
                photo,
                ticket_id
            )

            if storage_path is None:

                st.error(
                    "The photo could not be uploaded."
                )

            else:

                case_data = {
                    "id": case_id,
                    "ticket_id": ticket_id,
                    "name": name.strip(),
                    "age": int(age),
                    "gender": gender,
                    "location": location.strip(),
                    "last_seen": last_seen.strip(),
                    "reporter_name": reporter_name.strip(),
                    "contact": contact.strip(),
                    "description": description.strip(),
                    "photo_path": storage_path,
                    "status": "Missing",
                    "created_at": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    )
                }

                result = insert_case(
                    case_data
                )

                if result is None:

                    delete_photo(
                        storage_path
                    )

                else:

                    st.session_state.cases = load_cases()

                    st.success(
                        "Missing person report submitted successfully!"
                    )

                    st.success(
                        f"Your Ticket ID is: {ticket_id}"
                    )

                    st.warning(
                        "Please save your Ticket ID."
                    )


# ============================================================
# TRACK TICKET
# ============================================================

elif page == "🎫 Track Ticket":

    st.title(
        "🎫 Track Missing Person Ticket"
    )

    ticket = st.text_input(
        "Enter Ticket ID",
        placeholder="Example: MP-20260917-0001"
    )

    if st.button(
        "🔎 Track Case",
        use_container_width=True
    ):

        if not ticket.strip():

            st.warning(
                "Please enter a Ticket ID."
            )

        else:

            matching_cases = [
                case
                for case in st.session_state.cases
                if str(
                    case.get(
                        "ticket_id",
                        ""
                    )
                ).lower()
                == ticket.strip().lower()
            ]

            if not matching_cases:

                st.error(
                    "No case found with this Ticket ID."
                )

            else:

                case = matching_cases[0]

                st.success(
                    "Case found."
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        case.get(
                            "name",
                            "Unknown"
                        )
                    )

                    st.write(
                        f"**Ticket ID:** "
                        f"{case.get('ticket_id', '')}"
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
                        f"**Location:** "
                        f"{case.get('location', '')}"
                    )

                    st.write(
                        f"**Last Seen:** "
                        f"{case.get('last_seen', '')}"
                    )

                with col2:

                    status = case.get(
                        "status",
                        "Missing"
                    )

                    if str(
                        status
                    ).lower() == "found":

                        st.success(
                            f"Status: {status}"
                        )

                    else:

                        st.error(
                            f"Status: {status}"
                        )

                    photo_url = get_public_photo_url(
                        case.get(
                            "photo_path"
                        )
                    )

                    if photo_url:

                        st.image(
                            photo_url,
                            width=250
                        )


# ============================================================
# SEARCH CASES
# ============================================================

elif page == "🔍 Search Cases":

    st.title(
        "🔍 Search Missing Person Cases"
    )

    st.write(
        "Search using a person's name and location."
    )

    search_name = st.text_input(
        "Person Name",
        placeholder="Example: Rahul"
    )

    search_location = st.text_input(
        "Location",
        placeholder="Example: Hyderabad"
    )

    search_status = st.selectbox(
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

            case_name = str(
                case.get(
                    "name",
                    ""
                )
            ).lower()

            case_location = str(
                case.get(
                    "location",
                    ""
                )
            ).lower()

            case_status = str(
                case.get(
                    "status",
                    ""
                )
            ).lower()

            name_match = (
                not search_name.strip()
                or search_name.strip().lower()
                in case_name
            )

            location_match = (
                not search_location.strip()
                or search_location.strip().lower()
                in case_location
            )

            status_match = (
                search_status == "All"
                or search_status.lower()
                == case_status
            )

            if (
                name_match
                and location_match
                and status_match
            ):

                results.append(case)

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

                        photo_url = get_public_photo_url(
                            case.get(
                                "photo_path"
                            )
                        )

                        if photo_url:

                            st.image(
                                photo_url,
                                width=220
                            )

                    with col2:

                        st.subheader(
                            case.get(
                                "name",
                                "Unknown"
                            )
                        )

                        st.write(
                            f"**Ticket:** "
                            f"{case.get('ticket_id', '')}"
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
                            f"**Location:** "
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
                                f"{case.get('description')}"
                            )


# ============================================================
# AI FACE SEARCH
# ============================================================

elif page == "🤖 AI Face Search":

    st.title(
        "🤖 AI Face Search"
    )

    st.write(
        """
        Upload a photograph and TRACE-AI will compare
        the detected face with photographs stored in
        reported cases.
        """
    )

    st.info(
        """
        The new face-recognition system uses an OpenCV
        SFace feature model. The displayed value is a
        similarity score, NOT a guaranteed identity
        accuracy percentage.
        """
    )

    st.warning(
        """
        A face similarity result is only a screening
        result. Always verify the person's identity
        before taking action.
        """
    )

    search_photo = st.file_uploader(
        "Upload a face photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="ai_search_photo"
    )

    if search_photo is not None:

        file_bytes = search_photo.getvalue()

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        uploaded_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if uploaded_image is None:

            st.error(
                "Could not read the uploaded image."
            )

        else:

            st.image(
                cv2.cvtColor(
                    uploaded_image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Uploaded Image",
                width=300
            )

            if st.button(
                "🤖 Search Face",
                use_container_width=True
            ):

                with st.spinner(
                    "AI is comparing faces..."
                ):

                    matched_case, match_data, error_message = (
                        find_best_face_match(
                            uploaded_image,
                            st.session_state.cases
                        )
                    )

                if error_message:

                    st.error(
                        error_message
                    )

                elif matched_case is None:

                    st.warning(
                        "No matching case could be identified."
                    )

                else:

                    similarity = match_data[
                        "similarity"
                    ]

                    score = match_data[
                        "score"
                    ]

                    usable_photos = match_data[
                        "usable_photos"
                    ]

                    st.write(
                        f"Cases with usable face photos: "
                        f"**{usable_photos}**"
                    )

                    # ====================================================
                    # STRONG MATCH
                    # ====================================================

                    if similarity >= FACE_MATCH_THRESHOLD:

                        st.success(
                            "Potential face match found."
                        )

                        st.metric(
                            "AI Similarity Score",
                            f"{score}%"
                        )

                        st.warning(
                            """
                            This is a potential match, not proof
                            of identity. Please verify the person
                            before taking action.
                            """
                        )

                        col1, col2 = st.columns(
                            [1, 2]
                        )

                        with col1:

                            photo_url = get_public_photo_url(
                                matched_case.get(
                                    "photo_path"
                                )
                            )

                            if photo_url:

                                st.image(
                                    photo_url,
                                    width=250
                                )

                        with col2:

                            st.subheader(
                                matched_case.get(
                                    "name",
                                    "Unknown"
                                )
                            )

                            st.write(
                                f"**Ticket:** "
                                f"{matched_case.get('ticket_id', '')}"
                            )

                            st.write(
                                f"**Age:** "
                                f"{matched_case.get('age', '')}"
                            )

                            st.write(
                                f"**Gender:** "
                                f"{matched_case.get('gender', '')}"
                            )

                            st.write(
                                f"**Location:** "
                                f"{matched_case.get('location', '')}"
                            )

                            st.write(
                                f"**Status:** "
                                f"{matched_case.get('status', '')}"
                            )

                        # =================================================
                        # REPORTER CONTACT
                        # =================================================

                        st.divider()

                        st.subheader(
                            "📞 Reporter Contact"
                        )

                        reporter_contact = matched_case.get(
                            "contact",
                            ""
                        )

                        if reporter_contact:

                            st.success(
                                f"📞 Reporter Contact Number: "
                                f"{reporter_contact}"
                            )

                        else:

                            st.warning(
                                "No reporter contact number "
                                "was provided."
                            )

                        # =================================================
                        # CONTACT ADMINISTRATOR
                        # =================================================

                        st.divider()

                        st.subheader(
                            "📧 Contact TRACE-AI Administrator"
                        )

                        st.write(
                            """
                            If you believe you have found this
                            person, please inform the TRACE-AI
                            administrator and provide the Ticket ID.
                            """
                        )

                        email_subject = (
                            "TRACE-AI Potential Match - "
                            + str(
                                matched_case.get(
                                    "ticket_id",
                                    ""
                                )
                            )
                        )

                        email_body = (
                            "Hello TRACE-AI Administrator,\n\n"
                            "I believe I may have found the "
                            "person associated with this case.\n\n"
                            "Ticket ID: "
                            + str(
                                matched_case.get(
                                    "ticket_id",
                                    ""
                                )
                            )
                            + "\n"
                            "Name: "
                            + str(
                                matched_case.get(
                                    "name",
                                    ""
                                )
                            )
                            + "\n"
                            "Location: "
                            + str(
                                matched_case.get(
                                    "location",
                                    ""
                                )
                            )
                            + "\n"
                            "AI Similarity Score: "
                            + str(score)
                            + "%\n\n"
                            "Please verify this information."
                        )

                        mailto_link = (
                            "mailto:"
                            + ADMIN_EMAIL
                            + "?subject="
                            + urllib.parse.quote(
                                email_subject
                            )
                            + "&body="
                            + urllib.parse.quote(
                                email_body
                            )
                        )

                        st.link_button(
                            "📧 Email Administrator",
                            mailto_link,
                            use_container_width=True
                        )

                        st.info(
                            f"Administrator Email: "
                            f"{ADMIN_EMAIL}"
                        )

                    # ====================================================
                    # NOT A STRONG MATCH
                    # ====================================================

                    else:

                        st.warning(
                            "No strong face match was found."
                        )

                        st.metric(
                            "Best Similarity Score",
                            f"{score}%"
                        )

                        st.write(
                            """
                            The closest stored face did not
                            reach the matching threshold.
                            Try a clear, front-facing photo
                            with good lighting.
                            """
                        )


# ============================================================
# LOCATION
# ============================================================

elif page == "📍 GPS Location":

    st.title(
        "📍 Location"
    )

    st.write(
        """
        This section uses the normal location name.
        Latitude and longitude are not required.
        """
    )

    if not st.session_state.admin_logged_in:

        st.warning(
            "Only the administrator can update case locations."
        )

    else:

        cases = st.session_state.cases

        if not cases:

            st.info(
                "No cases are available."
            )

        else:

            case_options = {
                (
                    f"{case.get('ticket_id', '')} - "
                    f"{case.get('name', '')}"
                ): case
                for case in cases
            }

            selected_label = st.selectbox(
                "Select Case",
                list(
                    case_options.keys()
                )
            )

            selected_case = case_options[
                selected_label
            ]

            current_location = selected_case.get(
                "location",
                ""
            )

            new_location = st.text_input(
                "Location",
                value=str(
                    current_location
                ),
                placeholder="Example: Hyderabad"
            )

            if st.button(
                "📍 Update Location",
                use_container_width=True
            ):

                if not new_location.strip():

                    st.error(
                        "Please enter a location."
                    )

                else:

                    result = update_case_location(
                        selected_case["id"],
                        new_location.strip()
                    )

                    if result is not None:

                        st.success(
                            "Location updated successfully."
                        )

                        st.session_state.cases = load_cases()

                        st.rerun()


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.title(
        "🚨 Missing Person Alerts"
    )

    missing_cases = [
        case
        for case in st.session_state.cases
        if str(
            case.get(
                "status",
                ""
            )
        ).lower() == "missing"
    ]

    if not missing_cases:

        st.success(
            "There are currently no active "
            "missing-person alerts."
        )

    else:

        st.warning(
            f"{len(missing_cases)} active "
            "missing-person case(s)."
        )

        for case in missing_cases:

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [1, 3]
                )

                with col1:

                    photo_url = get_public_photo_url(
                        case.get(
                            "photo_path"
                        )
                    )

                    if photo_url:

                        st.image(
                            photo_url,
                            width=200
                        )

                with col2:

                    st.subheader(
                        case.get(
                            "name",
                            "Unknown"
                        )
                    )

                    st.write(
                        f"**Location:** "
                        f"{case.get('location', '')}"
                    )

                    st.write(
                        f"**Last Seen:** "
                        f"{case.get('last_seen', '')}"
                    )

                    st.write(
                        f"**Ticket:** "
                        f"{case.get('ticket_id', '')}"
                    )

                    if case.get(
                        "description"
                    ):

                        st.write(
                            f"**Description:** "
                            f"{case.get('description')}"
                        )


# ============================================================
# CONTACT ADMINISTRATOR
# ============================================================

elif page == "📧 Contact Administrator":

    st.title(
        "📧 Contact TRACE-AI Administrator"
    )

    st.write(
        """
        If you believe you have found a missing person,
        please contact the TRACE-AI administrator.
        """
    )

    st.subheader(
        "Administrator Email"
    )

    st.info(
        ADMIN_EMAIL
    )

    subject = st.text_input(
        "Subject",
        value="TRACE-AI Missing Person Information"
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Enter the Ticket ID and information "
            "about where you found the person."
        )
    )

    email_link = (
        "mailto:"
        + ADMIN_EMAIL
        + "?subject="
        + urllib.parse.quote(subject)
        + "&body="
        + urllib.parse.quote(message)
    )

    st.link_button(
        "📧 Email Administrator",
        email_link,
        use_container_width=True
    )

    st.caption(
        "Please provide the Ticket ID when contacting "
        "the administrator."
    )


# ============================================================
# ADMIN LOGIN
# ============================================================

elif page == "🔐 Admin Login":

    st.title(
        "🔐 Admin Login"
    )

    if st.session_state.admin_logged_in:

        st.success(
            "You are already logged in."
        )

    else:

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True
        ):

            if (
                username == ADMIN_USER
                and password == ADMIN_PASSWORD
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
# ADMIN DASHBOARD
# ============================================================

if st.session_state.admin_logged_in:

    st.sidebar.divider()

    st.sidebar.subheader(
        "👨‍💼 Administration"
    )

    open_admin = st.sidebar.checkbox(
        "Open Admin Dashboard"
    )

    if open_admin:

        st.title(
            "👨‍💼 Admin Dashboard"
        )

        st.write(
            "Manage reported missing-person cases."
        )

        cases = st.session_state.cases

        if not cases:

            st.info(
                "No cases have been reported yet."
            )

        else:

            st.write(
                f"Total cases: **{len(cases)}**"
            )

            for case in cases:

                with st.container(
                    border=True
                ):

                    col1, col2 = st.columns(
                        [1, 3]
                    )

                    with col1:

                        photo_url = get_public_photo_url(
                            case.get(
                                "photo_path"
                            )
                        )

                        if photo_url:

                            st.image(
                                photo_url,
                                width=220
                            )

                    with col2:

                        st.subheader(
                            case.get(
                                "name",
                                "Unknown"
                            )
                        )

                        st.write(
                            f"**Ticket ID:** "
                            f"{case.get('ticket_id', '')}"
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
                            f"**Location:** "
                            f"{case.get('location', '')}"
                        )

                        st.write(
                            f"**Last Seen:** "
                            f"{case.get('last_seen', '')}"
                        )

                        st.write(
                            f"**Reporter:** "
                            f"{case.get('reporter_name', '')}"
                        )

                        st.success(
                            f"📞 Reporter Contact: "
                            f"{case.get('contact', 'Not provided')}"
                        )

                        st.write(
                            f"**Description:** "
                            f"{case.get('description', '')}"
                        )

                        st.write(
                            f"**Current Status:** "
                            f"{case.get('status', '')}"
                        )

                        current_status = str(
                            case.get(
                                "status",
                                "Missing"
                            )
                        )

                        status_index = 0

                        if (
                            current_status.lower()
                            == "found"
                        ):

                            status_index = 1

                        new_status = st.selectbox(
                            "Change Status",
                            [
                                "Missing",
                                "Found"
                            ],
                            index=status_index,
                            key=(
                                f"status_"
                                f"{case.get('id')}"
                            )
                        )

                        update_col, delete_col = st.columns(2)

                        with update_col:

                            if st.button(
                                "💾 Update Status",
                                key=(
                                    f"update_"
                                    f"{case.get('id')}"
                                ),
                                use_container_width=True
                            ):

                                result = update_case_status(
                                    case.get("id"),
                                    new_status
                                )

                                if result is not None:

                                    st.session_state.cases = load_cases()

                                    st.success(
                                        "Status updated successfully."
                                    )

                                    st.rerun()

                        with delete_col:

                            if st.button(
                                "🗑️ Delete Case",
                                key=(
                                    f"delete_"
                                    f"{case.get('id')}"
                                ),
                                use_container_width=True
                            ):

                                deleted = delete_case(
                                    case.get("id")
                                )

                                if deleted is not None:

                                    delete_photo(
                                        case.get(
                                            "photo_path"
                                        )
                                    )

                                    st.session_state.cases = load_cases()

                                    st.success(
                                        "Case deleted successfully."
                                    )

                                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TRACE-AI • Finding Missing People Using AI"
)
