const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const resultDiv = document.getElementById('result');
const probsDiv = document.getElementById('probs');
const uploadPreview = document.getElementById('uploadPreview');

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCamBtn = document.getElementById('startCam');
const snapBtn = document.getElementById('snap');
const autoBtn = document.getElementById('auto');

let autoMode = false;
let mediaStream=null;
let autoInterval=null;

uploadForm.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  const imgURL = URL.createObjectURL(fileInput.files[0]);
  uploadPreview.innerHTML = `<img src="${imgURL}" style="max-width:100%;border-radius:12px;">`

  const r = await fetch('/predict',{method:'POST',body:fd});
  const j = await r.json();
  show(j);
});

startCamBtn.addEventListener('click', async ()=>{
  mediaStream = await navigator.mediaDevices.getUserMedia({video:true});
  video.srcObject = mediaStream;
});

snapBtn.addEventListener('click', ()=>{ shotPredict(); });

autoBtn.addEventListener('click', ()=>{
  autoMode = !autoMode;
  autoBtn.textContent = autoMode?"Auto: ON":"Auto Predict";
  if(autoMode){
    autoInterval=setInterval(shotPredict,1500);
  }else clearInterval(autoInterval);
});

function shotPredict(){
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0);
  fetch("/predict",{
    method:"POST",
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({imageBase64:canvas.toDataURL()})
  }).then(r=>r.json()).then(show)
}

function show(res){
  resultDiv.innerHTML = `Prediction: <span class="badge">${res.class}</span>`;
  probsDiv.innerHTML="";
  Object.entries(res.probs).forEach(([k,v])=>{
    const div=document.createElement('div');
    div.innerHTML = `<b>${k}</b> ${(v*100).toFixed(1)}% <div class="progress"><span style="width:${v*100}%"></span></div>`
    probsDiv.appendChild(div);
  })
}
