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

st.set_page_config(page_title="TRACE-AI", page_icon="🔎", layout="wide")

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
ADMIN_USER = "admin"
ADMIN_PASSWORD = "Swarajyam@2014"
ADMIN_EMAIL = "rkotha2@student.gitam.edu"

# SFace cosine threshold. This is deliberately stricter than the old 0.50.
# It is a similarity threshold, NOT identity accuracy.
FACE_MATCH_THRESHOLD = 0.75

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
        storage_path = f"cases/{ticket_id}_{uuid4().hex[:10]}.{extension}"
        supabase.storage.from_(BUCKET_NAME).upload(
            storage_path,
            uploaded_file.getvalue(),
            {"content-type": uploaded_file.type or "image/jpeg", "upsert": "false"},
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
        return supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
    except Exception:
        return None


def download_image(storage_path):
    url = get_public_photo_url(storage_path)
    if not url:
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

# ============================================================
# FACE MODELS
# ============================================================
@st.cache_resource
def load_yunet():
    """Download and cache YuNet, a robust face detector with landmarks."""
    model_url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    model_path = os.path.join(tempfile.gettempdir(), "traceai_yunet_2023mar.onnx")
    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
            req = urllib.request.Request(model_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(model_path, "wb") as f:
                f.write(data)
        detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), 0.65, 0.3, 5000
        )
        return detector
    except Exception:
        return None


@st.cache_resource
def load_sface_model():
    """Download and cache OpenCV SFace."""
    model_url = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx"
    )
    model_path = os.path.join(tempfile.gettempdir(), "traceai_sface.onnx")
    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
            req = urllib.request.Request(model_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            with open(model_path, "wb") as f:
                f.write(data)
        return cv2.FaceRecognizerSF_create(model_path, "")
    except Exception:
        return None


@st.cache_resource
def load_haar_detectors():
    """Fallback Haar detectors for installations where YuNet cannot load."""
    frontal = None
    profile = None
    try:
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        frontal = cv2.CascadeClassifier(frontal_path)
        if frontal.empty():
            frontal = None
    except Exception:
        frontal = None
    try:
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        profile = cv2.CascadeClassifier(profile_path)
        if profile.empty():
            profile = None
    except Exception:
        profile = None
    return frontal, profile


# ============================================================
# IMAGE / FACE DETECTION
# ============================================================
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def prepare_image(image):
    if image is None:
        return None
    image = image.copy()
    h, w = image.shape[:2]
    max_side = 1400
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
        # Highest detector confidence, with area as a tie-breaker.
        best = max(faces, key=lambda f: (float(f[14]), float(f[2] * f[3])))
        return image, best.astype(np.float32)
    except Exception:
        return None


def detect_with_haar(image, frontal, profile):
    if image is None:
        return None
    image = prepare_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    candidates = []
    for cascade, scale_factor, neighbors in (
        (frontal, 1.05, 4),
        (frontal, 1.08, 3),
        (profile, 1.05, 3),
    ):
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
                candidates.append((int(x), int(y), int(w), int(h)))
        except Exception:
            pass

    if not candidates:
        return None
    x, y, w, h = max(candidates, key=lambda b: b[2] * b[3])
    return image, (x, y, w, h)


def find_face(image):
    """Try YuNet, then multiple rotated Haar passes."""
    if image is None:
        return None

    image = prepare_image(image)
    yunet = load_yunet()

    # YuNet is much better at non-perfectly-frontal faces.
    for angle in (0, -15, 15, -30, 30):
        rotated = rotate_image(image, angle) if angle else image
        result = detect_with_yunet(rotated, yunet)
        if result is not None:
            return {"method": "yunet", "image": result[0], "face": result[1], "angle": angle}

    # Fallback for environments where YuNet is unavailable.
    frontal, profile = load_haar_detectors()
    best = None
    best_area = 0
    for angle in (0, -10, 10, -20, 20, -30, 30):
        rotated = rotate_image(image, angle) if angle else image
        result = detect_with_haar(rotated, frontal, profile)
        if result is not None:
            _, box = result
            area = box[2] * box[3]
            if area > best_area:
                best_area = area
                best = {"method": "haar", "image": result[0], "face": box, "angle": angle}
    return best


def extract_sface_feature(image, recognizer):
    if image is None or recognizer is None:
        return None
    detection = find_face(image)
    if detection is None:
        return None

    try:
        work = detection["image"]
        face = detection["face"]

        if detection["method"] == "yunet":
            # YuNet gives x,y,w,h and 5 landmarks. SFace's alignCrop uses
            # exactly this 15-value face record.
            aligned = recognizer.alignCrop(work, face)
            feature = recognizer.feature(aligned)
        else:
            x, y, w, h = face
            mx = int(w * 0.18)
            my = int(h * 0.22)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(work.shape[1], x + w + mx)
            y2 = min(work.shape[0], y + h + my)
            crop = work[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            feature = recognizer.feature(crop)

        if feature is None:
            return None
        feature = np.asarray(feature, dtype=np.float32).flatten()
        norm = np.linalg.norm(feature)
        if norm <= 1e-8:
            return None
        return feature / norm
    except Exception:
        return None


def cosine_similarity(feature1, feature2):
    if feature1 is None or feature2 is None:
        return -1.0
    try:
        a = np.asarray(feature1, dtype=np.float32).flatten()
        b = np.asarray(feature2, dtype=np.float32).flatten()
        if len(a) != len(b):
            return -1.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-8:
            return -1.0
        return float(np.dot(a, b) / denom)
    except Exception:
        return -1.0


def similarity_to_percentage(similarity):
    """Display-only score. It is not identity accuracy."""
    similarity = max(-1.0, min(1.0, float(similarity)))
    score = ((similarity - 0.20) / 0.80) * 100.0
    return round(max(0.0, min(100.0, score)), 1)


def find_best_face_match(uploaded_image, cases):
    recognizer = load_sface_model()
    if recognizer is None:
        return None, None, "The OpenCV SFace model could not be loaded. Please try again."

    # Check the uploaded image separately so the error is specifically useful.
    if find_face(uploaded_image) is None:
        return None, None, (
            "No usable face was detected in the uploaded photo. "
            "The detector tried several angles and lighting adjustments. "
            "Please use a clearer photo with the face larger in the frame."
        )

    uploaded_feature = extract_sface_feature(uploaded_image, recognizer)
    if uploaded_feature is None:
        return None, None, "The uploaded face could not be processed. Please try another clear photo."

    best_case = None
    best_similarity = -1.0
    usable_photos = 0

    for case in cases:
        storage_path = case.get("photo_path")
        if not storage_path:
            continue
        stored_image = download_image(storage_path)
        if stored_image is None:
            continue
        stored_feature = extract_sface_feature(stored_image, recognizer)
        if stored_feature is None:
            continue
        usable_photos += 1
        similarity = cosine_similarity(uploaded_feature, stored_feature)
        if similarity > best_similarity:
            best_similarity = similarity
            best_case = case

    if best_case is None:
        return None, None, "No usable case photographs were available for face comparison."

    return (
        best_case,
        {
            "similarity": best_similarity,
            "score": similarity_to_percentage(best_similarity),
            "usable_photos": usable_photos,
        },
        None,
    )


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
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.rerun()

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.title("🔎 TRACE-AI")
    st.subheader("Finding Missing People Using Artificial Intelligence")
    st.write(
        "TRACE-AI helps store and search missing-person cases using case information, "
        "photographs, location names and AI-based face comparison."
    )
    st.divider()
    total_cases = len(st.session_state.cases)
    missing_cases = sum(str(c.get("status", "")).lower() == "missing" for c in st.session_state.cases)
    found_cases = sum(str(c.get("status", "")).lower() == "found" for c in st.session_state.cases)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", total_cases)
    c2.metric("Missing", missing_cases)
    c3.metric("Found", found_cases)
    st.divider()
    st.info("Use the sidebar to report, track, search, or manage cases.")

# ============================================================
# REPORT
# ============================================================
elif page == "📝 Report Missing Person":
    st.title("📝 Report Missing Person")
    with st.form("report_form"):
        name = st.text_input("Full Name *")
        age = st.number_input("Age", min_value=0, max_value=120, value=18)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not Specified"])
        location = st.text_input("Last Known Location *", placeholder="Example: Hyderabad")
        last_seen = st.text_input("Last Seen Date / Time", placeholder="Example: 17 September 2026, 6:30 PM")
        reporter_name = st.text_input("Reporter's Name")
        contact = st.text_input("Reporter Contact Number *")
        description = st.text_area("Additional Description")
        photo = st.file_uploader("Upload Photo *", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("🚨 Submit Missing Person Report", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("Please enter the person's name.")
        elif not location.strip():
            st.error("Please enter the last known location.")
        elif not contact.strip():
            st.error("Please enter the reporter's contact number.")
        elif photo is None:
            st.error("Please upload a photo.")
        else:
            case_id = get_next_id()
            ticket_id = create_ticket_id(case_id)
            storage_path = upload_photo(photo, ticket_id)
            if storage_path is None:
                st.error("The photo could not be uploaded.")
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
                    "created_at": datetime.now().astimezone().isoformat(),
                }
                result = insert_case(case_data)
                if result is None:
                    delete_photo(storage_path)
                else:
                    st.session_state.cases = load_cases()
                    st.success("Missing person report submitted successfully!")
                    st.success(f"Your Ticket ID is: {ticket_id}")
                    st.warning("Please save your Ticket ID.")

# ============================================================
# TRACK
# ============================================================
elif page == "🎫 Track Ticket":
    st.title("🎫 Track Missing Person Ticket")
    ticket = st.text_input("Enter Ticket ID", placeholder="Example: MP-20260917-0001")
    if st.button("🔎 Track Case", use_container_width=True):
        if not ticket.strip():
            st.warning("Please enter a Ticket ID.")
        else:
            matches = [c for c in st.session_state.cases if str(c.get("ticket_id", "")).lower() == ticket.strip().lower()]
            if not matches:
                st.error("No case found with this Ticket ID.")
            else:
                case = matches[0]
                st.success("Case found.")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(case.get("name", "Unknown"))
                    st.write(f"**Ticket ID:** {case.get('ticket_id', '')}")
                    st.write(f"**Age:** {case.get('age', '')}")
                    st.write(f"**Gender:** {case.get('gender', '')}")
                    st.write(f"**Location:** {case.get('location', '')}")
                    st.write(f"**Last Seen:** {case.get('last_seen', '')}")
                with c2:
                    status = case.get("status", "Missing")
                    if str(status).lower() == "found":
                        st.success(f"Status: {status}")
                    else:
                        st.error(f"Status: {status}")
                    url = get_public_photo_url(case.get("photo_path"))
                    if url:
                        st.image(url, width=250)

# ============================================================
# SEARCH CASES
# ============================================================
elif page == "🔍 Search Cases":
    st.title("🔍 Search Missing Person Cases")
    st.write("Search using a person's name and location.")
    search_name = st.text_input("Person Name", placeholder="Example: Rahul")
    search_location = st.text_input("Location", placeholder="Example: Hyderabad")
    search_status = st.selectbox("Status", ["All", "Missing", "Found"])
    if st.button("🔎 Search", use_container_width=True):
        results = []
        for case in st.session_state.cases:
            name_match = not search_name.strip() or search_name.strip().lower() in str(case.get("name", "")).lower()
            location_match = not search_location.strip() or search_location.strip().lower() in str(case.get("location", "")).lower()
            status_match = search_status == "All" or search_status.lower() == str(case.get("status", "")).lower()
            if name_match and location_match and status_match:
                results.append(case)
        if not results:
            st.warning("No matching cases found.")
        else:
            st.success(f"{len(results)} case(s) found.")
            for case in results:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        url = get_public_photo_url(case.get("photo_path"))
                        if url:
                            st.image(url, width=220)
                    with c2:
                        st.subheader(case.get("name", "Unknown"))
                        st.write(f"**Ticket:** {case.get('ticket_id', '')}")
                        st.write(f"**Age:** {case.get('age', '')}")
                        st.write(f"**Gender:** {case.get('gender', '')}")
                        st.write(f"**Location:** {case.get('location', '')}")
                        st.write(f"**Last Seen:** {case.get('last_seen', '')}")
                        st.write(f"**Status:** {case.get('status', '')}")
                        if case.get("description"):
                            st.write(f"**Description:** {case.get('description')}")

# ============================================================
# AI FACE SEARCH
# ============================================================
elif page == "🤖 AI Face Search":
    st.title("🤖 AI Face Search")
    st.write("Upload a photograph and TRACE-AI will compare the detected face with photographs stored in reported cases.")
    st.info(
        "TRACE-AI uses OpenCV YuNet/SFace face detection and recognition. "
        "The displayed percentage is a similarity score, NOT guaranteed identity accuracy."
    )
    st.warning("A face similarity result is only a screening result. Always verify identity before taking action.")

    search_photo = st.file_uploader(
        "Upload a face photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="ai_search_photo",
    )

    if search_photo is not None:
        image_array = np.frombuffer(search_photo.getvalue(), dtype=np.uint8)
        uploaded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if uploaded_image is None:
            st.error("Could not read the uploaded image.")
        else:
            st.image(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)
            if st.button("🤖 Search Face", use_container_width=True):
                with st.spinner("AI is detecting and comparing faces..."):
                    matched_case, match_data, error_message = find_best_face_match(
                        uploaded_image, st.session_state.cases
                    )

                if error_message:
                    st.error(error_message)
                elif matched_case is None:
                    st.warning("No matching case could be identified.")
                else:
                    similarity = match_data["similarity"]
                    score = match_data["score"]
                    usable_photos = match_data["usable_photos"]
                    st.write(f"Cases with usable face photos: **{usable_photos}**")

                    if similarity >= FACE_MATCH_THRESHOLD:
                        st.success("Potential face match found.")
                        st.metric("AI Similarity Score", f"{score}%")
                        st.warning(
                            "This is a potential match, not proof of identity. "
                            "Please verify the person before taking action."
                        )
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            url = get_public_photo_url(matched_case.get("photo_path"))
                            if url:
                                st.image(url, width=250)
                        with c2:
                            st.subheader(matched_case.get("name", "Unknown"))
                            st.write(f"**Ticket:** {matched_case.get('ticket_id', '')}")
                            st.write(f"**Age:** {matched_case.get('age', '')}")
                            st.write(f"**Gender:** {matched_case.get('gender', '')}")
                            st.write(f"**Location:** {matched_case.get('location', '')}")
                            st.write(f"**Status:** {matched_case.get('status', '')}")

                        st.divider()
                        st.subheader("📞 Reporter Contact")
                        reporter_contact = matched_case.get("contact", "")
                        if reporter_contact:
                            st.success(f"📞 Reporter Contact Number: {reporter_contact}")
                        else:
                            st.warning("No reporter contact number was provided.")

                        st.divider()
                        st.subheader("📧 Contact TRACE-AI Administrator")
                        st.write("If you believe you have found this person, inform the TRACE-AI administrator and provide the Ticket ID.")
                        email_subject = "TRACE-AI Potential Match - " + str(matched_case.get("ticket_id", ""))
                        email_body = (
                            "Hello TRACE-AI Administrator,\n\n"
                            "I believe I may have found the person associated with this case.\n\n"
                            f"Ticket ID: {matched_case.get('ticket_id', '')}\n"
                            f"Name: {matched_case.get('name', '')}\n"
                            f"Location: {matched_case.get('location', '')}\n"
                            f"AI Similarity Score: {score}%\n\n"
                            "Please verify this information."
                        )
                        mailto = (
                            f"mailto:{ADMIN_EMAIL}?subject={urllib.parse.quote(email_subject)}"
                            f"&body={urllib.parse.quote(email_body)}"
                        )
                        st.link_button("📧 Email Administrator", mailto, use_container_width=True)
                        st.info(f"Administrator Email: {ADMIN_EMAIL}")
                    else:
                        st.warning("No strong face match was found.")
                        st.metric("Best Similarity Score", f"{score}%")
                        st.write(
                            "The closest stored face did not reach the stricter matching threshold. "
                            "A result below the threshold is not treated as a potential match."
                        )

# ============================================================
# LOCATION
# ============================================================
elif page == "📍 GPS Location":
    st.title("📍 Location")
    st.write("This section uses the normal location name. Latitude and longitude are not required.")
    if not st.session_state.admin_logged_in:
        st.warning("Only the administrator can update case locations.")
    else:
        cases = st.session_state.cases
        if not cases:
            st.info("No cases are available.")
        else:
            options = {f"{c.get('ticket_id', '')} - {c.get('name', '')}": c for c in cases}
            label = st.selectbox("Select Case", list(options.keys()))
            selected = options[label]
            new_location = st.text_input(
                "Location",
                value=str(selected.get("location", "")),
                placeholder="Example: Hyderabad",
            )
            if st.button("📍 Update Location", use_container_width=True):
                if not new_location.strip():
                    st.error("Please enter a location.")
                else:
                    result = update_case_location(selected["id"], new_location.strip())
                    if result is not None:
                        st.success("Location updated successfully.")
                        st.session_state.cases = load_cases()
                        st.rerun()

# ============================================================
# ALERTS
# ============================================================
elif page == "🚨 Alerts":
    st.title("🚨 Missing Person Alerts")
    missing_cases = [c for c in st.session_state.cases if str(c.get("status", "")).lower() == "missing"]
    if not missing_cases:
        st.success("There are currently no active missing-person alerts.")
    else:
        st.warning(f"{len(missing_cases)} active missing-person case(s).")
        for case in missing_cases:
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    url = get_public_photo_url(case.get("photo_path"))
                    if url:
                        st.image(url, width=200)
                with c2:
                    st.subheader(case.get("name", "Unknown"))
                    st.write(f"**Location:** {case.get('location', '')}")
                    st.write(f"**Last Seen:** {case.get('last_seen', '')}")
                    st.write(f"**Ticket:** {case.get('ticket_id', '')}")
                    if case.get("description"):
                        st.write(f"**Description:** {case.get('description')}")

# ============================================================
# CONTACT ADMIN
# ============================================================
elif page == "📧 Contact Administrator":
    st.title("📧 Contact TRACE-AI Administrator")
    st.write("If you believe you have found a missing person, please contact the TRACE-AI administrator.")
    st.subheader("Administrator Email")
    st.info(ADMIN_EMAIL)
    subject = st.text_input("Subject", value="TRACE-AI Missing Person Information")
    message = st.text_area(
        "Message",
        placeholder="Enter the Ticket ID and information about where you found the person.",
    )
    email_link = (
        f"mailto:{ADMIN_EMAIL}?subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(message)}"
    )
    st.link_button("📧 Email Administrator", email_link, use_container_width=True)
    st.caption("Please provide the Ticket ID when contacting the administrator.")

# ============================================================
# ADMIN LOGIN
# ============================================================
elif page == "🔐 Admin Login":
    st.title("🔐 Admin Login")
    if st.session_state.admin_logged_in:
        st.success("You are already logged in.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔐 Login", use_container_width=True):
            if username == ADMIN_USER and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

# ============================================================
# ADMIN DASHBOARD
# ============================================================
if st.session_state.admin_logged_in:
    st.sidebar.divider()
    st.sidebar.subheader("👨‍💼 Administration")
    open_admin = st.sidebar.checkbox("Open Admin Dashboard")
    if open_admin:
        st.title("👨‍💼 Admin Dashboard")
        st.write("Manage reported missing-person cases.")
        cases = st.session_state.cases
        if not cases:
            st.info("No cases have been reported yet.")
        else:
            st.write(f"Total cases: **{len(cases)}**")
            for case in cases:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        url = get_public_photo_url(case.get("photo_path"))
                        if url:
                            st.image(url, width=220)
                    with c2:
                        st.subheader(case.get("name", "Unknown"))
                        st.write(f"**Ticket ID:** {case.get('ticket_id', '')}")
                        st.write(f"**Age:** {case.get('age', '')}")
                        st.write(f"**Gender:** {case.get('gender', '')}")
                        st.write(f"**Location:** {case.get('location', '')}")
                        st.write(f"**Last Seen:** {case.get('last_seen', '')}")
                        st.write(f"**Reporter:** {case.get('reporter_name', '')}")
                        st.success(f"📞 Reporter Contact: {case.get('contact', 'Not provided')}")
                        st.write(f"**Description:** {case.get('description', '')}")
                        st.write(f"**Current Status:** {case.get('status', '')}")

                        current_status = str(case.get("status", "Missing"))
                        status_index = 1 if current_status.lower() == "found" else 0
                        new_status = st.selectbox(
                            "Change Status",
                            ["Missing", "Found"],
                            index=status_index,
                            key=f"status_{case.get('id')}",
                        )
                        update_col, delete_col = st.columns(2)
                        with update_col:
                            if st.button("💾 Update Status", key=f"update_{case.get('id')}", use_container_width=True):
                                result = update_case_status(case.get("id"), new_status)
                                if result is not None:
                                    st.session_state.cases = load_cases()
                                    st.success("Status updated successfully.")
                                    st.rerun()
                        with delete_col:
                            if st.button("🗑️ Delete Case", key=f"delete_{case.get('id')}", use_container_width=True):
                                deleted = delete_case(case.get("id"))
                                if deleted is not None:
                                    delete_photo(case.get("photo_path"))
                                    st.session_state.cases = load_cases()
                                    st.success("Case deleted successfully.")
                                    st.rerun()

st.divider()
st.caption("TRACE-AI • Finding Missing People Using AI")
