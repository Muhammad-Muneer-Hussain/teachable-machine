import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# Global Configuration
BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Teachable Machine Replica", page_icon="🤖", layout="wide")
st.title("Teachable Machine")
st.markdown("Train your own machine learning model right in the browser, completely offline!")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

st.header("1. Gather Data")

col1, col2 = st.columns(2)

def get_webcam_html(class_name):
    return f"""
    <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <video id="video_{class_name}" width="100%" autoplay playsinline style="border-radius: 8px; background: #000; transform: scaleX(-1);"></video>
        <button id="capture_{class_name}" style="width: 100%; padding: 12px; margin-top: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; user-select: none;">Hold to Record</button>
        <p id="status_{class_name}" style="margin-top: 8px; margin-bottom: 0; font-size: 14px; color: #555; text-align: center;"></p>
        <canvas id="canvas_{class_name}" width="224" height="224" style="display:none;"></canvas>
    </div>
    <script>
        const video = document.getElementById('video_{class_name}');
        const canvas = document.getElementById('canvas_{class_name}');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status_{class_name}');
        const btn = document.getElementById('capture_{class_name}');
        
        navigator.mediaDevices.getUserMedia({{ video: true }})
            .then(stream => {{ video.srcObject = stream; }})
            .catch(err => {{ status.innerText = "Camera access denied."; }});
            
        btn.addEventListener('mousedown', startCapturing);
        btn.addEventListener('mouseup', stopCapturing);
        btn.addEventListener('mouseleave', stopCapturing);
        btn.addEventListener('touchstart', startCapturing);
        btn.addEventListener('touchend', stopCapturing);
        
        let captureInterval = null;
        let count = 0;
        let isUploading = false;
        
        async function sendFrame() {{
            if (isUploading) return;
            isUploading = true;
            
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -224, 0, 224, 224);
            ctx.restore();
            
            let blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            let formData = new FormData();
            formData.append('class_name', '{class_name}');
            formData.append('files', new File([blob], 'capture.jpg', {{ type: 'image/jpeg' }}));
            
            try {{
                let res = await fetch('{BACKEND_URL}/upload-sample', {{ method: 'POST', body: formData }});
                if (res.ok) {{
                    count++;
                    status.innerText = `Uploaded ${{count}} image samples.`;
                }}
            }} catch(err) {{
                status.innerText = "Backend connection error.";
            }}
            isUploading = false;
        }}

        function startCapturing(e) {{
            e.preventDefault();
            btn.style.backgroundColor = "#45a049";
            btn.innerText = "Recording...";
            sendFrame();
            captureInterval = setInterval(sendFrame, 150);
        }}
        
        function stopCapturing(e) {{
            e.preventDefault();
            btn.style.backgroundColor = "#4CAF50";
            btn.innerText = "Hold to Record";
            if (captureInterval) clearInterval(captureInterval);
            captureInterval = null;
        }}
    </script>
    """

with col1:
    class_1 = st.text_input("Class 1 Name", value="Class 1")
    components.html(get_webcam_html(class_1), height=420)

with col2:
    class_2 = st.text_input("Class 2 Name", value="Class 2")
    components.html(get_webcam_html(class_2), height=420)


st.markdown("---")
st.header("2. Train Model")

if st.button("🚀 Train Model", use_container_width=True):
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

if st.session_state.model_trained:
    st.markdown("---")
    st.header("3. Live Prediction")
    st.markdown("Real-time webcam inference without clicking buttons!")
    
    live_pred_html = f"""
    <div style="font-family: sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 12px; background: #fff; max-width: 600px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <video id="pred_video" width="100%" autoplay playsinline style="border-radius: 8px; background: #000; transform: scaleX(-1);"></video>
        <canvas id="pred_canvas" width="224" height="224" style="display:none;"></canvas>
        <div id="pred_result" style="font-size: 26px; font-weight: bold; margin-top: 20px; text-align: center; color: #222;">Waiting for camera...</div>
        <div id="all_probs" style="margin-top: 20px;"></div>
    </div>
    <script>
        const video = document.getElementById('pred_video');
        const canvas = document.getElementById('pred_canvas');
        const ctx = canvas.getContext('2d');
        const resultDiv = document.getElementById('pred_result');
        const probsDiv = document.getElementById('all_probs');
        let isPredicting = false;
        
        navigator.mediaDevices.getUserMedia({{ video: true }})
            .then(stream => {{ 
                video.srcObject = stream; 
                resultDiv.innerText = "Connecting to model...";
                setInterval(predictFrame, 300); // Send frame every 300ms
            }})
            .catch(err => {{ resultDiv.innerText = "Camera access denied."; }});
            
        async function predictFrame() {{
            if (isPredicting) return;
            isPredicting = true;
            
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -224, 0, 224, 224);
            ctx.restore();
            
            let blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            let formData = new FormData();
            formData.append('file', new File([blob], 'frame.jpg', {{ type: 'image/jpeg' }}));
            
            try {{
                let res = await fetch('{BACKEND_URL}/predict', {{ method: 'POST', body: formData }});
                if (res.ok) {{
                    let data = await res.json();
                    resultDiv.innerText = `Prediction: ${{data.class}}`;
                    
                    let probsHtml = '';
                    for (let [cls, prob] of Object.entries(data.all_probabilities)) {{
                        probsHtml += `
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <span style="width: 120px; font-weight: bold; font-size: 16px; color: #444; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${{cls}}</span>
                                <div style="flex-grow: 1; background-color: #e0e0e0; height: 24px; border-radius: 12px; overflow: hidden; margin-left: 10px; position: relative;">
                                    <div style="width: ${{prob}}%; background-color: #ff9800; height: 100%; transition: width 0.2s;"></div>
                                </div>
                                <span style="width: 60px; text-align: right; margin-left: 10px; font-size: 14px; font-weight: bold; color: #333;">${{prob.toFixed(0)}}%</span>
                            </div>
                        `;
                    }}
                    probsDiv.innerHTML = probsHtml;
                }}
            }} catch(err) {{
                // Ignore silent fail
            }}
            isPredicting = false;
        }}
    </script>
    """
    components.html(live_pred_html, height=700)
