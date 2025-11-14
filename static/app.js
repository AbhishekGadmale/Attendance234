// Basic camera + capture + API calls
const video = document.getElementById('video');
let streamHandle = null;
let recognizeInterval = null;

async function startCamera(){
  try{
    streamHandle = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = streamHandle;
    await video.play();
  }catch(e){
    alert('Camera access denied or not available.');
  }
}

function stopCamera(){
  if(streamHandle){
    streamHandle.getTracks().forEach(t=>t.stop());
    streamHandle = null;
  }
}

function captureImage(){
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function captureAndSave(){
  const uname = document.getElementById('uname').value.trim();
  const uid = document.getElementById('uid').value.trim();
  const uclass = document.getElementById('class_select_reg').value;
  if(!uname || !uid){ alert('Enter name and roll'); return; }
  // capture nimgs times
  for(let i=0;i<10;i++){
    const img = captureImage();
    await fetch('/add_user_api',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ username: uname, userid: uid, image: img }) });
    await new Promise(r=>setTimeout(r,300));
  }
  alert('Images saved & model will train if enough images');
}

async function recognizeOnce(){
  const img = captureImage();
  const cls = document.getElementById('class_select_live').value;
  const res = await fetch('/recognize_api',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image: img, class: cls }) });
  const j = await res.json();
  const out = document.getElementById('recognizeResult');
  if(j.found){ out.innerText = 'Recognized: ' + j.name; }
  else out.innerText = 'No face detected';
}

function startRecognizeLoop(){
  if(!streamHandle) startCamera();
  recognizeInterval = setInterval(recognizeOnce, 2000);
}
function stopRecognizeLoop(){ if(recognizeInterval){ clearInterval(recognizeInterval); recognizeInterval=null; } }

// Buttons
document.getElementById('captureBtn').addEventListener('click', ()=>{ captureAndSave(); });
document.getElementById('recognizeBtn').addEventListener('click', ()=>{ startRecognizeLoop(); });
document.getElementById('stopBtn').addEventListener('click', ()=>{ stopRecognizeLoop(); });

// Start camera on load
startCamera();
