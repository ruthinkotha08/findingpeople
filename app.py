import streamlit as st
import cv2
import numpy as np
import urllib.request
import urllib.parse
import tempfile
import os
import hashlib
from datetime import datetime
from uuid import uuid4
from supabase import create_client, Client

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="TRACE-AI",
    page_icon="🔎",
    layout="wide",
)

# ============================================================
# SUPABASE / SETTINGS
# ============================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Supabase connection could not be established.")
    st.error(str(e))
    st.stop()

BUCKET_NAME = "case-photos"

# Keep administrator credentials in Streamlit Secrets.
# Add ADMIN_USER and ADMIN_PASSWORD to Secrets.
ADMIN_USER = st.secrets.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = traceai123
ADMIN_EMAIL = "rkotha2@student.gitam.edu"

# IMPORTANT:
# SFace uses a cosine similarity value, not a percentage.
# Use the OpenCV SFace cosine reference threshold for the same-person decision.
SFACE_REFERENCE_THRESHOLD = 0.363
FACE_MATCH_THRESHOLD = 0.363
SECOND_VIEW_THRESHOLD = 0.30
ENSEMBLE_THRESHOLD = 0.363

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "cases" not in st.session_state:
    st.session_state.cases = []

# ============================================================
# DATABASE
# ============================================================
def load_cases():
    try:
        response = (
            supabase.table("cases")
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
            supabase.table("cases")
            .select("id")
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return int(response.data[0]["id"]) + 1
    except Exception:
        pass
    return 1


def create_ticket_id(case_id):
    return f"MP-{datetime.now().strftime('%Y%m%d')}-{case_id:04d}"


def insert_case(case_data):
    try:
        response = supabase.table("cases").insert(case_data).execute()
        return response.data
    except Exception as e:
        st.error("Could not save the case.")
        st.error(str(e))
        return None


def update_case_status(case_id, new_status):
    try:
        response = (
            supabase.table("cases")
            .update({"status": new_status})
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
            supabase.table("cases")
            .update({"location": new_location})
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
            supabase.table("cases")
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
        extension = uploaded_file.name.split(".")[-1].lower()
        if extension not in {"jpg", "jpeg", "png", "webp"}:
            extension = "jpg"

        storage_path = (
            f"cases/{ticket_id}_{uuid4().hex[:10]}.{extension}"
        )

        supabase.storage.from_(BUCKET_NAME).upload(
            storage_path,
            uploaded_file.getvalue(),
            {
                "content-type": uploaded_file.type or "image/jpeg",
                "upsert": "false",
            },
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
        supabase.storage.from_(BUCKET_NAME).remove([storage_path])
    except Exception:
        pass


def get_public_photo_url(storage_path):
    if not storage_path:
        return None

    try:
        return supabase.storage.from_(BUCKET_NAME).get_public_url(
            storage_path
        )
    except Exception:
        return None


def download_image_bytes(storage_path):
    url = get_public_photo_url(storage_path)

    if not url:
        return None

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    except Exception:
        return None


def image_from_bytes(data):
    if not data:
        return None

    try:
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def sha256_bytes(data):
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()

# ============================================================
# FACE MODELS
# ============================================================
@st.cache_resource
def load_yunet():
    """Download and cache the OpenCV YuNet face detector."""
    model_url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    model_path = os.path.join(
        tempfile.gettempdir(),
        "traceai_yunet_2023mar.onnx",
    )

    try:
        if (
            not os.path.exists(model_path)
            or os.path.getsize(model_path) < 10000
        ):
            request = urllib.request.Request(
                model_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()

            with open(model_path, "wb") as f:
                f.write(data)

        return cv2.FaceDetectorYN_create(
            model_path,
            "",
            (320, 320),
            0.50,
            0.30,
            5000,
        )

    except Exception:
        return None


@st.cache_resource
def load_sface_model():
    """Load the official OpenCV SFace model."""
    model_url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    )

    model_path = os.path.join(
        tempfile.gettempdir(),
        "traceai_sface_2021dec.onnx",
    )

    try:
        if (
            not os.path.exists(model_path)
            or os.path.getsize(model_path) < 1000000
        ):
            request = urllib.request.Request(
                model_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()

            with open(model_path, "wb") as f:
                f.write(data)

        return cv2.FaceRecognizerSF_create(model_path, "")

    except Exception:
        return None


@st.cache_resource
def load_haar_detectors():
    frontal = None
    profile = None

    try:
        frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        if frontal.empty():
            frontal = None
    except Exception:
        frontal = None

    try:
        profile = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_profileface.xml"
        )
        if profile.empty():
            profile = None
    except Exception:
        profile = None

    return frontal, profile

# ============================================================
# IMAGE HELPERS
# ============================================================
def prepare_image(image):
    if image is None:
        return None

    image = image.copy()

    if len(image.shape) != 3:
        return image

    h, w = image.shape[:2]
    max_side = 1600

    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    return image


def make_enhanced_image(image):
    if image is None:
        return None

    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)

        enhanced = cv2.merge(
            (l_channel, a_channel, b_channel)
        )

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2BGR,
        )

    except Exception:
        return image


def detect_with_yunet(image, detector):
    if image is None or detector is None:
        return None

    try:
        image = prepare_image(image)
        h, w = image.shape[:2]

        detector.setInputSize((w, h))
        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        valid = []

        for face in faces:
            x, y, fw, fh = face[:4]

            if fw < 40 or fh < 40:
                continue

            confidence = float(face[14])
            area = float(fw * fh)

            valid.append(
                (
                    confidence,
                    area,
                    face.astype(np.float32),
                )
            )

        if not valid:
            return None

        # Prefer detection confidence, then face size.
        valid.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return image, valid[0][2]

    except Exception:
        return None


def detect_with_haar(image, frontal, profile):
    if image is None:
        return None

    image = prepare_image(image)

    try:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.equalizeHist(gray)

    except Exception:
        return None

    candidates = []

    settings = [
        (frontal, 1.03, 4),
        (frontal, 1.05, 4),
        (frontal, 1.08, 3),
        (frontal, 1.10, 3),
        (profile, 1.05, 4),
        (profile, 1.08, 3),
    ]

    for cascade, scale_factor, neighbors in settings:
        if cascade is None:
            continue

        try:
            boxes = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=neighbors,
                minSize=(50, 50),
            )

            for x, y, w, h in boxes:
                candidates.append(
                    (
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                    )
                )

        except Exception:
            pass

    if not candidates:
        return None

    # The largest face is normally the subject.
    box = max(
        candidates,
        key=lambda b: b[2] * b[3],
    )

    return image, box


def find_face(image):
    """
    Face detection:
    1. YuNet original
    2. YuNet enhanced
    3. Haar fallback

    We intentionally do not rotate the image before recognition.
    Rotating can cause the recognizer to compare a transformed
    face crop rather than the original aligned face.
    """
    if image is None:
        return None

    image = prepare_image(image)
    yunet = load_yunet()

    for version_name, version_image in (
        ("original", image),
        ("enhanced", make_enhanced_image(image)),
    ):
        result = detect_with_yunet(
            version_image,
            yunet,
        )

        if result is not None:
            return {
                "method": "yunet",
                "image": result[0],
                "face": result[1],
                "version": version_name,
            }

    frontal, profile = load_haar_detectors()

    for version_image in (
        image,
        make_enhanced_image(image),
    ):
        result = detect_with_haar(
            version_image,
            frontal,
            profile,
        )

        if result is not None:
            return {
                "method": "haar",
                "image": result[0],
                "face": result[1],
                "version": "haar",
            }

    return None

# ============================================================
# SFACE FEATURE EXTRACTION
# ============================================================
def get_aligned_face(image, recognizer):
    detection = find_face(image)

    if detection is None:
        return None

    try:
        work = detection["image"]
        face = detection["face"]

        if detection["method"] == "yunet":
            aligned = recognizer.alignCrop(work, face)

            if aligned is None or aligned.size == 0:
                return None

            return aligned

        # Haar has no five-point landmarks. Use a square crop as a
        # fallback only; YuNet is preferred for SFace recognition.
        x, y, w, h = [int(v) for v in face]
        size = max(w, h)
        cx = x + w // 2
        cy = y + h // 2

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(work.shape[1], x1 + size)
        y2 = min(work.shape[0], y1 + size)

        crop = work[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

    except Exception:
        return None


def extract_sface_features(image, recognizer):
    """Extract SFace features from a small set of controlled views."""
    if image is None or recognizer is None:
        return []

    aligned = get_aligned_face(image, recognizer)

    if aligned is None:
        return []

    views = [aligned]

    # A horizontal flip helps with small left/right framing differences.
    views.append(cv2.flip(aligned, 1))

    # Mild illumination normalization.
    try:
        lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )
        views.append(enhanced)
    except Exception:
        pass

    features = []

    for view in views:
        try:
            feature = recognizer.feature(view)
            if feature is not None and feature.size:
                # Keep the original OpenCV feature matrix. Do not replace
                # it with a hand-made normalized dot product.
                features.append(feature.copy())
        except Exception:
            pass

    return features


def sface_cosine(feature1, feature2, recognizer):
    """Use OpenCV's own FaceRecognizerSF.match() implementation."""
    if feature1 is None or feature2 is None or recognizer is None:
        return -1.0

    try:
        return float(
            recognizer.match(
                feature1,
                feature2,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )
    except Exception:
        return -1.0


def compare_feature_sets(features1, features2, recognizer):
    """
    Compare SFace feature matrices using OpenCV's official cosine
    matcher. We use the strongest and second-strongest view scores,
    but never invent a score from a custom dot-product formula.
    """
    scores = []

    for f1 in features1:
        for f2 in features2:
            score = sface_cosine(f1, f2, recognizer)
            if score >= 0.0:
                scores.append(score)

    if not scores:
        return {
            "best": -1.0,
            "second": -1.0,
            "ensemble": -1.0,
        }

    scores.sort(reverse=True)

    best = scores[0]
    second = scores[1] if len(scores) > 1 else best

    # Keep the two strongest scores for diagnostics.
    # The BEST score is used for the identity decision because the
    # official OpenCV SFace threshold applies to a direct feature pair.

    return {
        "best": float(best),
        "second": float(second),
        "ensemble": float((best + second) / 2.0),
    }


# Project display thresholds.
# IMPORTANT: these are decision/display thresholds, not scientific
# probabilities of identity.
SAME_PERSON_THRESHOLD = 0.363
SIMILAR_FACE_THRESHOLD = 0.250


def display_match_result(similarity, exact_photo=False):
    """
    Project display rule:

    - Exact same file -> 100% MATCH.
    - Facial similarity at/above SAME_PERSON_THRESHOLD ->
      100% MATCH (the system classifies it as the same-person match).
    - Lower but above SIMILAR_FACE_THRESHOLD -> show the actual
      similarity score as "similar face features".
    - Below SIMILAR_FACE_THRESHOLD -> 0% / no useful match.

    The displayed 100% means "the matching system classified this
    case as a same-person match"; it is NOT a scientific accuracy
    probability.
    """
    if exact_photo:
        return {
            "display_score": 100.0,
            "result_type": "same",
            "is_match": True,
        }

    if similarity is None or similarity < 0:
        return {
            "display_score": 0.0,
            "result_type": "none",
            "is_match": False,
        }

    similarity = max(0.0, min(1.0, float(similarity)))

    if similarity >= SAME_PERSON_THRESHOLD:
        return {
            "display_score": 100.0,
            "result_type": "same",
            "is_match": True,
        }

    if similarity >= SIMILAR_FACE_THRESHOLD:
        return {
            "display_score": round(similarity * 100.0, 1),
            "result_type": "similar",
            "is_match": False,
        }

    return {
        "display_score": 0.0,
        "result_type": "none",
        "is_match": False,
    }


def sha256_bytes(data):
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


def exact_photo_match(uploaded_bytes, stored_bytes):
    if uploaded_bytes is None or stored_bytes is None:
        return False

    uploaded_hash = sha256_bytes(uploaded_bytes)
    stored_hash = sha256_bytes(stored_bytes)

    return (
        uploaded_hash is not None
        and stored_hash is not None
        and uploaded_hash == stored_hash
    )


# ============================================================
# FACE SEARCH
# ============================================================
def find_best_face_match(uploaded_image, uploaded_bytes, cases):
    """
    Search every stored case using the official OpenCV SFace cosine
    matcher. Exact file equality is handled separately.

    Matching uses OpenCV SFace cosine similarity.

    The official OpenCV example uses cosine >= 0.363 as its
    same-identity reference threshold. We therefore use the BEST
    valid SFace view for the same-person decision instead of forcing
    an average of two views to cross a higher threshold. The other
    view scores remain visible for diagnostics.

    This is a prototype screening rule, not a probability of identity.
    """
    recognizer = load_sface_model()

    if recognizer is None:
        return (
            None,
            None,
            "The OpenCV SFace model could not be loaded. "
            "Please restart the app and try again.",
        )

    # Exact same uploaded file.
    for case in cases:
        storage_path = case.get("photo_path")
        if not storage_path:
            continue

        stored_bytes = download_image_bytes(storage_path)
        if stored_bytes is None:
            continue

        if exact_photo_match(uploaded_bytes, stored_bytes):
            return (
                case,
                {
                    "similarity": 1.0,
                    "score": 100.0,
                    "best": 1.0,
                    "second": 1.0,
                    "ensemble": 1.0,
                    "usable_photos": 1,
                    "exact_photo": True,
                    "result_type": "same",
                    "is_match": True,
                },
                None,
            )

    uploaded_features = extract_sface_features(
        uploaded_image,
        recognizer,
    )

    if not uploaded_features:
        return (
            None,
            None,
            "No usable face could be extracted from the uploaded "
            "photo. Please use a clear photo where the face is visible.",
        )

    best_case = None
    best_data = None
    usable_photos = 0

    for case in cases:
        storage_path = case.get("photo_path")
        if not storage_path:
            continue

        stored_image = download_image(storage_path)
        if stored_image is None:
            continue

        stored_features = extract_sface_features(
            stored_image,
            recognizer,
        )

        if not stored_features:
            continue

        usable_photos += 1

        comparison = compare_feature_sets(
            uploaded_features,
            stored_features,
            recognizer,
        )

        if (
            best_data is None
            or comparison["ensemble"] > best_data["ensemble"]
        ):
            best_case = case
            best_data = comparison

    if best_case is None:
        return (
            None,
            None,
            "No usable case photographs were available for face comparison.",
        )

    best = best_data["best"]
    second = best_data["second"]
    ensemble = best_data["ensemble"]

    display_result = display_match_result(
        best,
        exact_photo=False,
    )

    return (
        best_case,
        {
            "similarity": ensemble,
            "score": display_result["display_score"],
            "result_type": display_result["result_type"],
            "best": best,
            "second": second,
            "ensemble": ensemble,
            "usable_photos": usable_photos,
            "exact_photo": False,
            "is_match": display_result["is_match"],
        },
        None,
    )


# ============================================================
# IMAGE DOWNLOAD
# ============================================================
def download_image(storage_path):
    data = download_image_bytes(storage_path)

    if data is None:
        return None

    return image_from_bytes(data)

# ============================================================
# LOAD CASES
# ============================================================
st.session_state.cases = load_cases()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🔎 TRACE-AI")
st.sidebar.caption("Finding Missing People Using AI")
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
        "🔐 Admin Login",
    ],
)

if st.session_state.admin_logged_in:
    st.sidebar.success("Admin logged in")

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):
        st.session_state.admin_logged_in = False
        st.rerun()

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.title("🔎 TRACE-AI")
    st.subheader(
        "Finding Missing People Using Artificial Intelligence"
    )

    st.write(
        "TRACE-AI helps store and search missing-person "
        "cases using case information, photographs, "
        "location names and AI-based face comparison."
    )

    st.divider()

    total_cases = len(st.session_state.cases)

    missing_cases = sum(
        str(c.get("status", "")).lower() == "missing"
        for c in st.session_state.cases
    )

    found_cases = sum(
        str(c.get("status", "")).lower() == "found"
        for c in st.session_state.cases
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Cases", total_cases)
    c2.metric("Missing", missing_cases)
    c3.metric("Found", found_cases)

    st.divider()

    st.info(
        "Use the sidebar to report, track, search, "
        "or manage cases."
    )

# ============================================================
# REPORT
# ============================================================
elif page == "📝 Report Missing Person":
    st.title("📝 Report Missing Person")

    with st.form("report_form"):
        name = st.text_input("Full Name *")

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=18,
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other",
                "Not Specified",
            ],
        )

        location = st.text_input(
            "Last Known Location *",
            placeholder="Example: Hyderabad",
        )

        last_seen = st.text_input(
            "Last Seen Date / Time",
            placeholder="Example: 17 September 2026, 6:30 PM",
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
                "webp",
            ],
        )

        submitted = st.form_submit_button(
            "🚨 Submit Missing Person Report",
            use_container_width=True,
        )

    if submitted:
        if not name.strip():
            st.error("Please enter the person's name.")

        elif not location.strip():
            st.error(
                "Please enter the last known location."
            )

        elif not contact.strip():
            st.error(
                "Please enter the reporter's contact number."
            )

        elif photo is None:
            st.error("Please upload a photo.")

        else:
            case_id = get_next_id()
            ticket_id = create_ticket_id(case_id)

            storage_path = upload_photo(
                photo,
                ticket_id,
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
                    "created_at": datetime.now()
                    .astimezone()
                    .isoformat(),
                }

                result = insert_case(case_data)

                if result is None:
                    delete_photo(storage_path)

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
# TRACK
# ============================================================
elif page == "🎫 Track Ticket":
    st.title("🎫 Track Missing Person Ticket")

    ticket = st.text_input(
        "Enter Ticket ID",
        placeholder="Example: MP-20260917-0001",
    )

    if st.button(
        "🔎 Track Case",
        use_container_width=True,
    ):
        if not ticket.strip():
            st.warning("Please enter a Ticket ID.")

        else:
            matches = [
                c
                for c in st.session_state.cases
                if str(
                    c.get("ticket_id", "")
                ).lower()
                == ticket.strip().lower()
            ]

            if not matches:
                st.error(
                    "No case found with this Ticket ID."
                )

            else:
                case = matches[0]

                st.success("Case found.")

                c1, c2 = st.columns(2)

                with c1:
                    st.subheader(
                        case.get(
                            "name",
                            "Unknown",
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

                with c2:
                    status = case.get(
                        "status",
                        "Missing",
                    )

                    if str(status).lower() == "found":
                        st.success(
                            f"Status: {status}"
                        )
                    else:
                        st.error(
                            f"Status: {status}"
                        )

                    url = get_public_photo_url(
                        case.get("photo_path")
                    )

                    if url:
                        st.image(
                            url,
                            width=250,
                        )

# ============================================================
# SEARCH CASES
# ============================================================
elif page == "🔍 Search Cases":
    st.title("🔍 Search Missing Person Cases")

    st.write(
        "Search using a person's name and location."
    )

    search_name = st.text_input(
        "Person Name",
        placeholder="Example: Rahul",
    )

    search_location = st.text_input(
        "Location",
        placeholder="Example: Hyderabad",
    )

    search_status = st.selectbox(
        "Status",
        [
            "All",
            "Missing",
            "Found",
        ],
    )

    if st.button(
        "🔎 Search",
        use_container_width=True,
    ):
        results = []

        for case in st.session_state.cases:
            name_match = (
                not search_name.strip()
                or search_name.strip().lower()
                in str(
                    case.get("name", "")
                ).lower()
            )

            location_match = (
                not search_location.strip()
                or search_location.strip().lower()
                in str(
                    case.get("location", "")
                ).lower()
            )

            status_match = (
                search_status == "All"
                or search_status.lower()
                == str(
                    case.get("status", "")
                ).lower()
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
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])

                    with c1:
                        url = get_public_photo_url(
                            case.get("photo_path")
                        )

                        if url:
                            st.image(
                                url,
                                width=220,
                            )

                    with c2:
                        st.subheader(
                            case.get(
                                "name",
                                "Unknown",
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

                        if case.get("description"):
                            st.write(
                                f"**Description:** "
                                f"{case.get('description')}"
                            )

# ============================================================
# AI FACE SEARCH
# ============================================================
elif page == "🤖 AI Face Search":
    st.title("🤖 AI Face Search")

    st.write(
        "Upload a photograph and TRACE-AI will compare "
        "it with photographs stored in reported cases."
    )

    st.info(
        "The system first checks for the exact same photo. "
        "If it is a different photo, OpenCV YuNet + SFace "
        "compare the detected face using multiple controlled "
        "face views. The percentage is a similarity score, "
        "not identity accuracy."
    )

    st.warning(
        "A face similarity result is only a screening result. "
        "Always verify identity before taking action."
    )

    st.write(
        "Matching rule: OpenCV SFace cosine BEST score ≥ 0.363 "
        "is classified as a same-person match. Scores from 0.250 "
        "to 0.362 are shown as similar-face results; below 0.250 "
        "is shown as 0%."
    )

    search_photo = st.file_uploader(
        "Upload a face photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="ai_search_photo",
    )

    if search_photo is not None:
        uploaded_bytes = search_photo.getvalue()

        image_array = np.frombuffer(
            uploaded_bytes,
            dtype=np.uint8,
        )

        uploaded_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if uploaded_image is None:
            st.error(
                "Could not read the uploaded image."
            )

        else:
            st.image(
                cv2.cvtColor(
                    uploaded_image,
                    cv2.COLOR_BGR2RGB,
                ),
                caption="Uploaded Image",
                width=300,
            )

            if st.button(
                "🤖 Search Face",
                use_container_width=True,
            ):
                with st.spinner(
                    "AI is detecting and comparing faces..."
                ):
                    (
                        matched_case,
                        match_data,
                        error_message,
                    ) = find_best_face_match(
                        uploaded_image,
                        uploaded_bytes,
                        st.session_state.cases,
                    )

                if error_message:
                    st.error(error_message)

                elif matched_case is None:
                    st.warning(
                        "No matching case could be identified."
                    )

                else:
                    similarity = match_data["similarity"]
                    score = match_data["score"]
                    best_score = match_data["best"]
                    second_score = match_data["second"]
                    ensemble = match_data["ensemble"]
                    usable_photos = match_data["usable_photos"]
                    exact_photo = match_data.get(
                        "exact_photo",
                        False,
                    )
                    is_match = match_data.get(
                        "is_match",
                        False,
                    )

                    st.write(
                        "Cases with usable face photos: "
                        f"**{usable_photos}**"
                    )

                    result_type = match_data.get(
                        "result_type",
                        "none",
                    )

                    if exact_photo or result_type == "same":
                        st.success(
                            "✅ SAME PERSON MATCH"
                        )

                        st.metric(
                            "Match Accuracy",
                            "100%",
                        )

                        if not exact_photo:
                            st.caption(
                                "100% here means TRACE-AI classified "
                                "the facial comparison as a same-person "
                                "match. It is not a scientific probability "
                                "of identity."
                            )

                        with st.expander(
                            "Show technical SFace score"
                        ):
                            st.write(
                                f"Best cosine: "
                                f"{best_score:.4f}"
                            )
                            st.write(
                                f"Second cosine: "
                                f"{second_score:.4f}"
                            )
                            st.write(
                                f"Ensemble cosine: "
                                f"{ensemble:.4f}"
                            )

                    elif result_type == "similar":
                        st.warning(
                            "🟡 SIMILAR FACE FEATURES"
                        )

                        st.metric(
                            "Face Similarity",
                            f"{score}%",
                        )

                        st.write(
                            "The uploaded face has similar features "
                            "to the closest case, but it did not reach "
                            "the same-person threshold."
                        )

                        with st.expander(
                            "Show technical SFace score"
                        ):
                            st.write(
                                f"Best cosine: "
                                f"{best_score:.4f}"
                            )
                            st.write(
                                f"Second cosine: "
                                f"{second_score:.4f}"
                            )
                            st.write(
                                f"Ensemble cosine: "
                                f"{ensemble:.4f}"
                            )

                    else:
                        st.error(
                            "❌ NO SIGNIFICANT FACE MATCH"
                        )

                        st.metric(
                            "Face Similarity",
                            "0%",
                        )

                        with st.expander(
                            "Show technical SFace score"
                        ):
                            st.write(
                                f"Best cosine: "
                                f"{best_score:.4f}"
                            )
                            st.write(
                                f"Second cosine: "
                                f"{second_score:.4f}"
                            )
                            st.write(
                                f"Ensemble cosine: "
                                f"{ensemble:.4f}"
                            )

                    # ------------------------------------------------
                    # Only show case details/contact for a strong match.
                    # ------------------------------------------------
                    if is_match or exact_photo:
                        c1, c2 = st.columns([1, 2])

                        with c1:
                            url = get_public_photo_url(
                                matched_case.get("photo_path")
                            )

                            if url:
                                st.image(
                                    url,
                                    width=250,
                                )

                        with c2:
                            st.subheader(
                                matched_case.get(
                                    "name",
                                    "Unknown",
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

                        st.divider()

                        st.subheader(
                            "📞 Reporter Contact"
                        )

                        reporter_contact = matched_case.get(
                            "contact",
                            "",
                        )

                        if reporter_contact:
                            st.success(
                                "📞 Reporter Contact Number: "
                                f"{reporter_contact}"
                            )
                        else:
                            st.warning(
                                "No reporter contact number "
                                "was provided."
                            )

                        st.divider()

                        st.subheader(
                            "📧 Contact TRACE-AI Administrator"
                        )

                        st.write(
                            "If you believe you have found this "
                            "person, inform the TRACE-AI administrator "
                            "and provide the Ticket ID."
                        )

                        email_subject = (
                            "TRACE-AI Potential Match - "
                            + str(
                                matched_case.get(
                                    "ticket_id",
                                    "",
                                )
                            )
                        )

                        email_body = (
                            "Hello TRACE-AI Administrator,\n\n"
                            "I believe I may have found the person "
                            "associated with this case.\n\n"
                            f"Ticket ID: "
                            f"{matched_case.get('ticket_id', '')}\n"
                            f"Name: "
                            f"{matched_case.get('name', '')}\n"
                            f"Location: "
                            f"{matched_case.get('location', '')}\n"
                            f"Face Match Result: {score}%\n\n"
                            "Please verify this information."
                        )

                        mailto = (
                            f"mailto:{ADMIN_EMAIL}"
                            f"?subject="
                            f"{urllib.parse.quote(email_subject)}"
                            f"&body="
                            f"{urllib.parse.quote(email_body)}"
                        )

                        st.link_button(
                            "📧 Email Administrator",
                            mailto,
                            use_container_width=True,
                        )

                        st.info(
                            f"Administrator Email: "
                            f"{ADMIN_EMAIL}"
                        )

# ============================================================
# LOCATION
# ============================================================
elif page == "📍 GPS Location":
    st.title("📍 Location")

    st.write(
        "This section uses the normal location name. "
        "Latitude and longitude are not required."
    )

    if not st.session_state.admin_logged_in:
        st.warning(
            "Only the administrator can update case locations."
        )

    else:
        cases = st.session_state.cases

        if not cases:
            st.info("No cases are available.")

        else:
            options = {
                f"{c.get('ticket_id', '')} - "
                f"{c.get('name', '')}": c
                for c in cases
            }

            label = st.selectbox(
                "Select Case",
                list(options.keys()),
            )

            selected = options[label]

            new_location = st.text_input(
                "Location",
                value=str(
                    selected.get(
                        "location",
                        "",
                    )
                ),
                placeholder="Example: Hyderabad",
            )

            if st.button(
                "📍 Update Location",
                use_container_width=True,
            ):
                if not new_location.strip():
                    st.error(
                        "Please enter a location."
                    )

                else:
                    result = update_case_location(
                        selected["id"],
                        new_location.strip(),
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
    st.title("🚨 Missing Person Alerts")

    missing_cases = [
        c
        for c in st.session_state.cases
        if str(
            c.get("status", "")
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
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])

                with c1:
                    url = get_public_photo_url(
                        case.get("photo_path")
                    )

                    if url:
                        st.image(
                            url,
                            width=200,
                        )

                with c2:
                    st.subheader(
                        case.get(
                            "name",
                            "Unknown",
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

                    if case.get("description"):
                        st.write(
                            f"**Description:** "
                            f"{case.get('description')}"
                        )

# ============================================================
# CONTACT ADMIN
# ============================================================
elif page == "📧 Contact Administrator":
    st.title(
        "📧 Contact TRACE-AI Administrator"
    )

    st.write(
        "If you believe you have found a missing person, "
        "please contact the TRACE-AI administrator."
    )

    st.subheader("Administrator Email")
    st.info(ADMIN_EMAIL)

    subject = st.text_input(
        "Subject",
        value="TRACE-AI Missing Person Information",
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Enter the Ticket ID and information "
            "about where you found the person."
        ),
    )

    email_link = (
        f"mailto:{ADMIN_EMAIL}"
        f"?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(message)}"
    )

    st.link_button(
        "📧 Email Administrator",
        email_link,
        use_container_width=True,
    )

    st.caption(
        "Please provide the Ticket ID when contacting "
        "the administrator."
    )

# ============================================================
# ADMIN LOGIN
# ============================================================
elif page == "🔐 Admin Login":
    st.title("🔐 Admin Login")

    if st.session_state.admin_logged_in:
        st.success("You are already logged in.")

    else:
        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password",
        )

        if st.button(
            "🔐 Login",
            use_container_width=True,
        ):
            if (
                username.strip() == str(ADMIN_USER).strip()
                and password.strip() == str(ADMIN_PASSWORD).strip()
            ):
                st.session_state.admin_logged_in = True

                st.success("Login successful.")
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
    st.sidebar.subheader("👨‍💼 Administration")

    open_admin = st.sidebar.checkbox(
        "Open Admin Dashboard"
    )

    if open_admin:
        st.title("👨‍💼 Admin Dashboard")

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
                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])

                    with c1:
                        url = get_public_photo_url(
                            case.get("photo_path")
                        )

                        if url:
                            st.image(
                                url,
                                width=220,
                            )

                    with c2:
                        st.subheader(
                            case.get(
                                "name",
                                "Unknown",
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
                            "📞 Reporter Contact: "
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
                                "Missing",
                            )
                        )

                        status_index = (
                            1
                            if current_status.lower()
                            == "found"
                            else 0
                        )

                        new_status = st.selectbox(
                            "Change Status",
                            [
                                "Missing",
                                "Found",
                            ],
                            index=status_index,
                            key=(
                                f"status_"
                                f"{case.get('id')}"
                            ),
                        )

                        update_col, delete_col = st.columns(2)

                        with update_col:
                            if st.button(
                                "💾 Update Status",
                                key=(
                                    f"update_"
                                    f"{case.get('id')}"
                                ),
                                use_container_width=True,
                            ):
                                result = update_case_status(
                                    case.get("id"),
                                    new_status,
                                )

                                if result is not None:
                                    st.session_state.cases = (
                                        load_cases()
                                    )

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
                                use_container_width=True,
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

                                    st.session_state.cases = (
                                        load_cases()
                                    )

                                    st.success(
                                        "Case deleted successfully."
                                    )
                                    st.rerun()
