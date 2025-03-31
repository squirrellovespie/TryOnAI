let clothingImg = new Image();
let video, canvas, ctx, pose;

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    if (data.success) {
        const optionsDiv = document.getElementById('clothing-options');
        optionsDiv.innerHTML = '<h2>Choose an Item to Try On:</h2>';
        data.clothing_items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'clothing-item';
            div.innerHTML = `
                <img src="${item.url}" alt="${item.label}">
                <p>${item.label}</p>
                <button onclick="selectItem('${item.url}')">Try On</button>
            `;
            optionsDiv.appendChild(div);
        });
    } else {
        alert('Failed to process image: ' + data.message);
    }
});

function selectItem(url) {
    clothingImg.src = url;
    clothingImg.onload = () => {
        console.log("Selected clothing loaded:", url);
        document.getElementById('start-camera').disabled = false;
    };
}

document.getElementById('start-camera').addEventListener('click', () => {
    document.getElementById('video-container').style.display = 'block';
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video = document.getElementById('video');
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            initPose();
        };
    }).catch((err) => console.error('Camera access denied:', err));
});

function initPose() {
    pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });
    pose.onResults(drawClothing);
    const camera = new Camera(video, {
        onFrame: async () => {
            await pose.send({ image: video });
        },
        width: 640,
        height: 480
    });
    camera.start();
}

function drawClothing(results) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (results.poseLandmarks) {
        const leftShoulder = results.poseLandmarks[11];  // Left shoulder
        const rightShoulder = results.poseLandmarks[12]; // Right shoulder
        const leftHip = results.poseLandmarks[23];      // Left hip
        const rightHip = results.poseLandmarks[24];     // Right hip

        if (leftShoulder && rightShoulder && leftHip && rightHip) {
            const shoulderX1 = leftShoulder.x * width;
            const shoulderY1 = leftShoulder.y * height;
            const shoulderX2 = rightShoulder.x * width;
            const shoulderY2 = rightShoulder.y * height;
            const hipX1 = leftHip.x * width;
            const hipY1 = leftHip.y * height;
            const hipX2 = rightHip.x * width;
            const hipY2 = rightHip.y * height;

            const shoulderDistance = Math.sqrt(Math.pow(shoulderX2 - shoulderX1, 2) + Math.pow(shoulderY2 - shoulderY1, 2));
            const hipDistance = Math.sqrt(Math.pow(hipX2 - hipX1, 2) + Math.pow(hipY2 - hipY1, 2));
            const torsoHeight = Math.abs((shoulderY1 + shoulderY2) / 2 - (hipY1 + hipY2) / 2);

            const targetWidth = shoulderDistance * 1.2;
            const scale = targetWidth / clothingImg.width;
            const clothingHeight = clothingImg.height * scale;

            // Position based on shoulders (for tops) or hips (for pants)
            let drawX, drawY;
            if (clothingImg.src.includes('pants') || clothingImg.src.includes('skirt')) {
                drawX = (hipX1 + hipX2) / 2 - targetWidth / 2;
                drawY = (hipY1 + hipY2) / 2 - clothingHeight / 2;
            } else {
                drawX = (shoulderX1 + shoulderX2) / 2 - targetWidth / 2;
                drawY = (shoulderY1 + shoulderY2) / 2 - clothingHeight * 0.3;
            }

            console.log("Drawing at:", drawX, drawY, targetWidth, clothingHeight);
            ctx.drawImage(clothingImg, drawX, drawY, targetWidth, clothingHeight);
        }
    }
}