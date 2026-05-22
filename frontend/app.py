import streamlit as st
import requests
import streamlit.components.v1 as components
import os

# Global Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Teachable Machine", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("Teachable Machine")
st.markdown("Train your own machine learning model right in the browser, completely offline!")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if "num_classes" not in st.session_state:
    st.session_state.num_classes = 2

def get_class_card_html(card_id, default_name):
    return f"""
    <div style="font-family: sans-serif; background-color: white; border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <input type="text" id="class_name_{card_id}" value="{default_name}" style="font-size: 20px; font-weight: bold; border: none; outline: none; border-bottom: 2px solid transparent; width: 100%;" onfocus="this.style.borderBottom='2px solid #1a73e8'" onblur="this.style.borderBottom='2px solid transparent'">
            <span style="color: #888; font-size: 18px;"></span>
        </div>
        <div id="status_{card_id}" style="font-size: 14px; margin-bottom: 10px; color: #555; font-weight: bold;">0 Image Samples</div>
        
        <div style="display: flex; gap: 10px; align-items: stretch; overflow-x: auto; padding-bottom: 8px;" id="gallery_row_{card_id}">
            <div id="action_buttons_{card_id}" style="display: flex; gap: 10px; flex-shrink: 0;">
                <button id="webcam_btn_{card_id}" style="width: 80px; height: 80px; background-color: #e8f0fe; color: #1a73e8; border: none; border-radius: 8px; font-weight: 500; cursor: pointer; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <span style="font-size: 24px; margin-bottom: 4px;">🎥</span><span style="font-size: 12px;">Webcam</span>
                </button>
                <button id="upload_btn_{card_id}" style="width: 80px; height: 80px; background-color: #e8f0fe; color: #1a73e8; border: none; border-radius: 8px; font-weight: 500; cursor: pointer; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <span style="font-size: 24px; margin-bottom: 4px;">⬆️</span><span style="font-size: 12px;">Upload</span>
                </button>
            </div>
            
            <div id="thumbnails_container_{card_id}" style="display: flex; gap: 5px; overflow-x: auto; flex-grow: 1;">
            </div>
        </div>
        
        <div id="webcam_container_{card_id}" style="display: none; margin-top: 15px; text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px;">
            <video id="video_{card_id}" width="100%" autoplay playsinline style="border-radius: 8px; transform: scaleX(-1); background-color: #000; max-height: 220px; object-fit: cover;"></video>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button id="record_btn_{card_id}" style="flex: 3; background-color: #f1f3f4; border: 1px solid #dadce0; color: #3c4043; border-radius: 20px; padding: 10px; font-weight: bold; cursor: pointer; user-select: none; -webkit-user-select: none;">Hold to Record</button>
                <button id="close_webcam_{card_id}" style="flex: 1; background-color: white; border: 1px solid #dadce0; color: #d93025; border-radius: 20px; padding: 10px; font-weight: bold; cursor: pointer;">Close</button>
            </div>
            <canvas id="canvas_{card_id}" width="224" height="224" style="display:none;"></canvas>
        </div>
        
        <div id="upload_container_{card_id}" style="display: none; margin-top: 15px; text-align: center; border: 2px dashed #dadce0; padding: 30px; border-radius: 8px; background: #f8f9fa;">
            <input type="file" id="file_input_{card_id}" multiple accept="image/*" style="display:none;">
            <button id="trigger_file_{card_id}" style="background-color: #1a73e8; color: white; border: none; border-radius: 5px; padding: 10px 20px; cursor: pointer; font-weight: 500; font-size: 16px;">Choose images from your files</button>
            <div style="margin-top: 15px;">
                 <button id="close_upload_{card_id}" style="background-color: transparent; border: none; color: #555; text-decoration: underline; cursor: pointer;">Cancel</button>
            </div>
            <div id="upload_status_{card_id}" style="margin-top: 15px; font-size: 14px; font-weight: bold; color: #1a73e8;"></div>
        </div>
    </div>
    
    <script>
        (function() {{
            const cardId = '{card_id}';
            const backendUrl = '{BACKEND_URL}';
            
            const classNameInput = document.getElementById('class_name_' + cardId);
            const webcamBtn = document.getElementById('webcam_btn_' + cardId);
            const uploadBtn = document.getElementById('upload_btn_' + cardId);
            const actionButtons = document.getElementById('action_buttons_' + cardId);
            const webcamContainer = document.getElementById('webcam_container_' + cardId);
            const uploadContainer = document.getElementById('upload_container_' + cardId);
            
            const video = document.getElementById('video_' + cardId);
            const recordBtn = document.getElementById('record_btn_' + cardId);
            const closeWebcamBtn = document.getElementById('close_webcam_' + cardId);
            const closeUploadBtn = document.getElementById('close_upload_' + cardId);
            const statusDiv = document.getElementById('status_' + cardId);
            const canvas = document.getElementById('canvas_' + cardId);
            const ctx = canvas.getContext('2d');
            
            const fileInput = document.getElementById('file_input_' + cardId);
            const triggerFileBtn = document.getElementById('trigger_file_' + cardId);
            const uploadStatusDiv = document.getElementById('upload_status_' + cardId);
            const thumbnailsContainer = document.getElementById('thumbnails_container_' + cardId);
            
            let streamRef = null;
            let captureInterval = null;
            let count = 0;
            let isUploading = false;
            
            let currentClassName = classNameInput.value;
            classNameInput.addEventListener('input', (e) => {{
                currentClassName = e.target.value;
            }});
            
            webcamBtn.onclick = () => {{
                uploadContainer.style.display = 'none';
                navigator.mediaDevices.getUserMedia({{ video: true }})
                    .then(stream => {{
                        streamRef = stream;
                        video.srcObject = stream;
                        webcamContainer.style.display = 'block';
                    }})
                    .catch(err => {{ 
                        alert("Camera access denied or unavailable."); 
                    }});
            }};
            
            closeWebcamBtn.onclick = () => {{
                if (streamRef) {{
                    streamRef.getTracks().forEach(track => track.stop());
                    streamRef = null;
                }}
                webcamContainer.style.display = 'none';
            }};

            closeUploadBtn.onclick = () => {{
                uploadContainer.style.display = 'none';
                actionButtons.style.display = 'flex';
            }};
            
            async function sendFrame() {{
                if (isUploading) return;
                isUploading = true;
                
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -224, 0, 224, 224);
                ctx.restore();
                
                let blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
                let formData = new FormData();
                formData.append('class_name', currentClassName);
                formData.append('files', new File([blob], 'capture.jpg', {{ type: 'image/jpeg' }}));
                
                try {{
                    let res = await fetch(backendUrl + '/upload-sample', {{ method: 'POST', body: formData }});
                    if (res.ok) {{
                        count++;
                        statusDiv.innerText = count + " Image Samples";
                        
                        let img = document.createElement('img');
                        img.src = canvas.toDataURL('image/jpeg', 0.8);
                        img.style.width = '80px';
                        img.style.height = '80px';
                        img.style.objectFit = 'cover';
                        img.style.borderRadius = '8px';
                        img.style.flexShrink = '0';
                        thumbnailsContainer.appendChild(img);
                        thumbnailsContainer.scrollLeft = thumbnailsContainer.scrollWidth;
                    }}
                }} catch(err) {{
                    statusDiv.innerText = "Error uploading.";
                }}
                isUploading = false;
            }}
            
            function startCapturing(e) {{
                e.preventDefault();
                recordBtn.style.backgroundColor = "#e8f0fe";
                recordBtn.style.borderColor = "#1a73e8";
                recordBtn.style.color = "#1a73e8";
                recordBtn.innerText = "Recording...";
                sendFrame();
                captureInterval = setInterval(sendFrame, 150);
            }}
            
            function stopCapturing(e) {{
                e.preventDefault();
                recordBtn.style.backgroundColor = "#f1f3f4";
                recordBtn.style.borderColor = "#dadce0";
                recordBtn.style.color = "#3c4043";
                recordBtn.innerText = "Hold to Record";
                if (captureInterval) clearInterval(captureInterval);
                captureInterval = null;
            }}
            
            recordBtn.addEventListener('mousedown', startCapturing);
            recordBtn.addEventListener('mouseup', stopCapturing);
            recordBtn.addEventListener('mouseleave', stopCapturing);
            recordBtn.addEventListener('touchstart', startCapturing);
            recordBtn.addEventListener('touchend', stopCapturing);
            
            uploadBtn.onclick = () => {{
                webcamContainer.style.display = 'none';
                uploadContainer.style.display = 'block';
            }};
            
            triggerFileBtn.onclick = () => fileInput.click();
            
            fileInput.onchange = async () => {{
                if (!fileInput.files.length) return;
                
                let formData = new FormData();
                formData.append('class_name', currentClassName);
                for (let i = 0; i < fileInput.files.length; i++) {{
                    formData.append('files', fileInput.files[i]);
                }}
                
                try {{
                    uploadStatusDiv.innerText = "Uploading...";
                    let res = await fetch(backendUrl + '/upload-sample', {{ method: 'POST', body: formData }});
                    if (res.ok) {{
                        count += fileInput.files.length;
                        statusDiv.innerText = count + " Image Samples";
                        uploadStatusDiv.innerText = "Uploaded " + fileInput.files.length + " images successfully!";
                        
                        for (let i = 0; i < fileInput.files.length; i++) {{
                            let file = fileInput.files[i];
                            let reader = new FileReader();
                            reader.onload = (e) => {{
                                let img = document.createElement('img');
                                img.src = e.target.result;
                                img.style.width = '80px';
                                img.style.height = '80px';
                                img.style.objectFit = 'cover';
                                img.style.borderRadius = '8px';
                                img.style.flexShrink = '0';
                                thumbnailsContainer.appendChild(img);
                            }}
                            reader.readAsDataURL(file);
                        }}
                        setTimeout(() => thumbnailsContainer.scrollLeft = thumbnailsContainer.scrollWidth, 100);
                    }} else {{
                        uploadStatusDiv.innerText = "Upload failed.";
                    }}
                }} catch(err) {{
                    uploadStatusDiv.innerText = "Error uploading.";
                }}
            }};
        }})();
    </script>
    """

def get_preview_html():
    html_code = """
    <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: #fff; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        
        <div style="display:flex; gap:10px; margin-bottom:15px;">
            <button id="use_webcam_btn" style="flex:1; padding:10px; border:none; border-radius:8px; background:#1a73e8; color:white; font-weight:bold; cursor:pointer;">
                Webcam
            </button>

            <button id="use_upload_btn" style="flex:1; padding:10px; border:none; border-radius:8px; background:#34a853; color:white; font-weight:bold; cursor:pointer;">
                Upload
            </button>
        </div>

        <input type="file" id="upload_input" accept="image/*" style="display:none;">

        <video id="pred_video" width="100%" autoplay playsinline
            style="border-radius: 8px; background: #000; transform: scaleX(-1); max-height: 220px; object-fit: cover;">
        </video>

        <img id="uploaded_preview"
            style="display:none; width:100%; border-radius:8px; max-height:220px; object-fit:cover;" />

        <canvas id="pred_canvas" width="224" height="224" style="display:none;"></canvas>

        <div id="pred_result"
            style="font-size: 24px; font-weight: bold; margin-top: 15px; text-align: center; color: #222;">
            Waiting...
        </div>

        <div id="all_probs" style="margin-top: 15px;"></div>
    </div>

    <script>
        (function() {

            const backendUrl = '""" + BACKEND_URL + """';

            const webcamBtn = document.getElementById('use_webcam_btn');
            const uploadBtn = document.getElementById('use_upload_btn');
            const uploadInput = document.getElementById('upload_input');

            const video = document.getElementById('pred_video');
            const uploadedPreview = document.getElementById('uploaded_preview');

            const canvas = document.getElementById('pred_canvas');
            const ctx = canvas.getContext('2d');

            const resultDiv = document.getElementById('pred_result');
            const probsDiv = document.getElementById('all_probs');

            let streamRef = null;
            let predictInterval = null;
            let isPredicting = false;

            startWebcam();

            webcamBtn.onclick = () => {
                uploadedPreview.style.display = 'none';
                video.style.display = 'block';
                startWebcam();
            };

            uploadBtn.onclick = () => {
                uploadInput.click();
            };

            uploadInput.onchange = async () => {

                const file = uploadInput.files[0];

                if (!file) return;

                stopWebcam();

                video.style.display = 'none';

                uploadedPreview.src = URL.createObjectURL(file);
                uploadedPreview.style.display = 'block';

                predictUploadedImage(file);
            };

            function startWebcam() {

                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {

                        streamRef = stream;

                        video.srcObject = stream;

                        resultDiv.innerText = "Predicting...";

                        if (predictInterval) {
                            clearInterval(predictInterval);
                        }

                        predictInterval = setInterval(predictFrame, 300);
                    })
                    .catch(err => {
                        resultDiv.innerText = "Camera access denied.";
                    });
            }

            function stopWebcam() {

                if (streamRef) {
                    streamRef.getTracks().forEach(track => track.stop());
                    streamRef = null;
                }

                if (predictInterval) {
                    clearInterval(predictInterval);
                }
            }

            async function predictFrame() {

                if (isPredicting) return;

                isPredicting = true;

                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -224, 0, 224, 224);
                ctx.restore();

                let blob = await new Promise(resolve =>
                    canvas.toBlob(resolve, 'image/jpeg', 0.8)
                );

                sendPrediction(blob);

                isPredicting = false;
            }

            async function predictUploadedImage(file) {
                sendPrediction(file);
            }

            async function sendPrediction(fileBlob) {

                let formData = new FormData();

                formData.append(
                    'file',
                    new File([fileBlob], 'image.jpg', { type: 'image/jpeg' })
                );

                try {

                    let res = await fetch(
                        backendUrl + '/predict',
                        {
                            method: 'POST',
                            body: formData
                        }
                    );

                    if (res.ok) {

                        let data = await res.json();

                        resultDiv.innerText =
                            "Prediction: " + data.class;

                        let probsHtml = '';

                        for (let [cls, prob] of Object.entries(data.all_probabilities)) {

                            probsHtml += `
                                <div style="display:flex; align-items:center; margin-bottom:12px;">
                                    <span style="width:120px; font-weight:bold;">
                                        ${cls}
                                    </span>

                                    <div style="flex-grow:1; background:#e0e0e0; height:24px; border-radius:12px; overflow:hidden; margin-left:10px;">
                                        <div style="width:${prob}%; background:#1a73e8; height:100%;"></div>
                                    </div>

                                    <span style="width:60px; text-align:right; margin-left:10px;">
                                        ${prob.toFixed(0)}%
                                    </span>
                                </div>
                            `;
                        }

                        probsDiv.innerHTML = probsHtml;
                    }

                } catch(err) {

                    resultDiv.innerText = "Prediction failed.";
                }
            }

        })();
    </script>
    """
    return html_code

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    # Height badha kar 700 kar di h aur scrolling true h, ab buttons kabhi nahi chupenge
    components.html(get_class_card_html("1", "Class 1"), height=900, scrolling=True)
    components.html(get_class_card_html("2", "Class 2"), height=700, scrolling=True)
    
    st.markdown("""
        <div style="border: 2px dashed #dadce0; border-radius: 12px; padding: 15px; text-align: center; color: #5f6368; font-weight: bold; cursor: not-allowed; background: #f8f9fa;">
            ➕ Add a class (Coming Soon)
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <h3 style="margin-top: 0;">Training</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Train Model", use_container_width=True, type="primary"):
        with st.spinner("Training in progress..."):
            try:
                response = requests.post(f"{BACKEND_URL}/train")
                if response.status_code == 200:
                    st.session_state.model_trained = True
                    st.success(response.json()["message"])
                    st.balloons()
                else:
                    st.error(f"Error: {response.json().get('detail', response.text)}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

    st.markdown("""
    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-top: 20px; margin-bottom: 10px;">
        <h3 style="margin-top: 0;">Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_trained:
        components.html(get_preview_html(), height=650)
    else:
        st.info("You must train a model on the left before you can preview it here.")