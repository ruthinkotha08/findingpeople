import streamlit as st
import cv2
import numpy as np
import urllib.request
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
# SUPABASE CONNECTION
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

# AI result threshold
MATCH_THRESHOLD = 50.0


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

            return int(
                response.data[0]["id"]
            ) + 1

        return 1

    except Exception:

        return 1


def create_ticket_id(case_id):

    date_part = datetime.now().strftime(
        "%Y%m%d"
    )

    return (
        f"MP-{date_part}-{case_id:04d}"
    )


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


def update_case_status(
    case_id,
    new_status
):

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


def update_case_location(
    case_id,
    new_location
):

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

        st.error("Could not update the location.")
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
# SUPABASE STORAGE FUNCTIONS
# ============================================================

def upload_photo(
    uploaded_file,
    ticket_id
):

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

        file_bytes = (
            uploaded_file.getvalue()
        )

        supabase.storage.from_(
            BUCKET_NAME
        ).upload(
            storage_path,
            file_bytes,
            {
                "content-type":
                    uploaded_file.type,
                "upsert":
                    "false"
            }
        )

        return storage_path

    except Exception as e:

        st.error("Photo upload failed.")
        st.error(str(e))

        return None


def delete_photo(
    storage_path
):

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


def get_public_photo_url(
    storage_path
):

    if not storage_path:
        return None

    try:

        return (
            supabase
            .storage
            .from_(BUCKET_NAME)
            .get_public_url(
                storage_path
            )
        )

    except Exception:

        return None


def download_image(
    storage_path
):

    url = get_public_photo_url(
        storage_path
    )

    if not url:
        return None

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
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
# HAAR CASCADE FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    # --------------------------------------------------------
    # METHOD 1: OpenCV installed cascade
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # METHOD 2: Download official OpenCV cascade
    # --------------------------------------------------------

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
                "User-Agent":
                    "Mozilla/5.0"
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

            temp_file.write(
                xml_data
            )

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

                os.remove(
                    temp_path
                )

            except Exception:

                pass

    return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(
    image,
    detector
):

    if image is None:
        return None

    if detector is None:
        return None

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(
            gray
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:

            return None

        x, y, w, h = max(
            faces,
            key=lambda box:
                box[2] * box[3]
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

    except Exception:

        return None


# ============================================================
# CREATE AI FACE MODEL
# ============================================================

def create_face_model(
    cases
):

    if not hasattr(
        cv2,
        "face"
    ):

        return None, {}

    detector = load_face_detector()

    if detector is None:

        return None, {}

    training_faces = []
    labels = []
    label_to_case = {}

    label_number = 0

    for case in cases:

        storage_path = case.get(
            "photo_path"
        )

        if not storage_path:

            continue

        image = download_image(
            storage_path
        )

        if image is None:

            continue

        face = detect_face(
            image,
            detector
        )

        if face is None:

            continue

        training_faces.append(
            face
        )

        labels.append(
            label_number
        )

        label_to_case[
            label_number
        ] = case

        label_number += 1

    if len(training_faces) == 0:

        return None, {}

    try:

        model = (
            cv2
            .face
            .LBPHFaceRecognizer_create()
        )

        model.train(
            training_faces,
            np.array(labels)
        )

        return (
            model,
            label_to_case
        )

    except Exception:

        return None, {}


# ============================================================
# AI SCORE
# ============================================================

def calculate_ai_score(
    distance
):

    if distance is None:

        return 0.0

    score = (
        100.0
        - (float(distance) * 0.5)
    )

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
        ).lower()
        == "missing"
    ])

    found_cases = len([
        case
        for case in st.session_state.cases
        if str(
            case.get(
                "status",
                ""
            )
        ).lower()
        == "found"
    ])

    col1, col2, col3 = \
        st.columns(3)

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

    st.write(
        "Enter the person's information below."
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
            placeholder=(
                "Example: Hyderabad"
            )
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
            "Contact Number"
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

                    "id":
                        case_id,

                    "ticket_id":
                        ticket_id,

                    "name":
                        name.strip(),

                    "age":
                        int(age),

                    "gender":
                        gender,

                    "location":
                        location.strip(),

                    "last_seen":
                        last_seen.strip(),

                    "reporter_name":
                        reporter_name.strip(),

                    "contact":
                        contact.strip(),

                    "description":
                        description.strip(),

                    "photo_path":
                        storage_path,

                    "status":
                        "Missing",

                    "created_at":
                        datetime.now()
                        .astimezone()
                        .isoformat()
                }

                result = insert_case(
                    case_data
                )

                if result is None:

                    delete_photo(
                        storage_path
                    )

                else:

                    st.session_state.cases = \
                        load_cases()

                    st.success(
                        "Missing person report submitted successfully!"
                    )

                    st.success(
                        f"Your Ticket ID is: {ticket_id}"
                    )

                    st.warning(
                        "Please save this Ticket ID for tracking."
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
        placeholder=(
            "Example: MP-20260917-0001"
        )
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

                col1, col2 = \
                    st.columns(2)

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

                    photo_url = \
                        get_public_photo_url(
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
        placeholder=(
            "Example: Hyderabad"
        )
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

                    col1, col2 = \
                        st.columns([1, 2])

                    with col1:

                        photo_url = \
                            get_public_photo_url(
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
        "A similarity score of 50% or higher is treated "
        "as a potential match."
    )

    st.warning(
        "The AI score is a screening result, not proof "
        "of identity. Always verify the person."
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

        file_bytes = \
            search_photo.getvalue()

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

                detector = \
                    load_face_detector()

                if detector is None:

                    st.error(
                        "Haar Cascade face detector "
                        "could not be loaded."
                    )

                else:

                    uploaded_face = \
                        detect_face(
                            uploaded_image,
                            detector
                        )

                    if uploaded_face is None:

                        st.error(
                            "No face detected in the uploaded photo."
                        )

                        st.info(
                            "Use a clear, front-facing "
                            "photograph with good lighting."
                        )

                    else:

                        with st.spinner(
                            "AI is comparing faces..."
                        ):

                            model, label_to_case = \
                                create_face_model(
                                    st.session_state.cases
                                )

                        if model is None:

                            if not hasattr(
                                cv2,
                                "face"
                            ):

                                st.error(
                                    "OpenCV Face module is not available."
                                )

                            else:

                                st.warning(
                                    "No usable case photographs "
                                    "are available for comparison."
                                )

                        else:

                            try:

                                predicted_label, distance = \
                                    model.predict(
                                        uploaded_face
                                    )

                                matched_case = \
                                    label_to_case.get(
                                        predicted_label
                                    )

                                score = \
                                    calculate_ai_score(
                                        distance
                                    )

                                # ==========================================
                                # 50% OR HIGHER
                                # ==========================================

                                if (
                                    matched_case
                                    and score >= MATCH_THRESHOLD
                                ):

                                    st.success(
                                        "Potential face match found."
                                    )

                                    st.metric(
                                        "AI Similarity Score",
                                        f"{score}%"
                                    )

                                    st.warning(
                                        "This is a potential match "
                                        "and should be verified by "
                                        "the appropriate authorities "
                                        "or administrator."
                                    )

                                    col1, col2 = \
                                        st.columns([1, 2])

                                    with col1:

                                        photo_url = \
                                            get_public_photo_url(
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

                                    # ======================================
                                    # REPORTER CONTACT
                                    # ======================================

                                    st.divider()

                                    st.subheader(
                                        "📞 Reporter Contact"
                                    )

                                    reporter_contact = \
                                        matched_case.get(
                                            "contact",
                                            ""
                                        )

                                    if reporter_contact:

                                        st.success(
                                            f"📞 Contact Number: "
                                            f"{reporter_contact}"
                                        )

                                        st.info(
                                            """
                                            If you believe you have
                                            found this person, contact
                                            the reporter using the number
                                            above and also inform the
                                            TRACE-AI administrator.

                                            Please verify the person's
                                            identity before taking action.
                                            """
                                        )

                                    else:

                                        st.warning(
                                            "The reporter did not provide "
                                            "a contact number."
                                        )

                                        st.info(
                                            "Please contact the "
                                            "TRACE-AI administrator."
                                        )

                                else:

                                    st.warning(
                                        "No potential match reached "
                                        "the 50% threshold."
                                    )

                                    st.write(
                                        f"Best calculated score: "
                                        f"{score}%"
                                    )

                                    st.info(
                                        "Try a clearer face photograph "
                                        "with good lighting."
                                    )

                            except Exception as e:

                                st.error(
                                    "Face comparison failed."
                                )

                                st.error(
                                    str(e)
                                )


# ============================================================
# GPS LOCATION
# ============================================================

elif page == "📍 GPS Location":

    st.title(
        "📍 GPS Location"
    )

    st.write(
        """
        This section now uses the normal location name.
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
                ):
                case

                for case in cases
            }

            selected_label = st.selectbox(
                "Select Case",
                list(
                    case_options.keys()
                )
            )

            selected_case = \
                case_options[
                    selected_label
                ]

            current_location = \
                selected_case.get(
                    "location",
                    ""
                )

            new_location = st.text_input(
                "Location",
                value=str(
                    current_location
                ),
                placeholder=(
                    "Example: Hyderabad"
                )
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

                    result = \
                        update_case_location(
                            selected_case["id"],
                            new_location.strip()
                        )

                    if result is not None:

                        st.success(
                            "Location updated successfully."
                        )

                        st.session_state.cases = \
                            load_cases()

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
        ).lower()
        == "missing"
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

                col1, col2 = \
                    st.columns([1, 3])

                with col1:

                    photo_url = \
                        get_public_photo_url(
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

                    col1, col2 = \
                        st.columns([1, 3])

                    with col1:

                        photo_url = \
                            get_public_photo_url(
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

                        update_col, delete_col = \
                            st.columns(2)

                        with update_col:

                            if st.button(
                                "💾 Update Status",
                                key=(
                                    f"update_"
                                    f"{case.get('id')}"
                                ),
                                use_container_width=True
                            ):

                                result = \
                                    update_case_status(
                                        case.get("id"),
                                        new_status
                                    )

                                if result is not None:

                                    st.session_state.cases = \
                                        load_cases()

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

                                deleted = \
                                    delete_case(
                                        case.get("id")
                                    )

                                if deleted is not None:

                                    delete_photo(
                                        case.get(
                                            "photo_path"
                                        )
                                    )

                                    st.session_state.cases = \
                                        load_cases()

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
