// --- Page Navigation ---
const btnDetector = document.getElementById('btn-detector');
const btnAnalysis = document.getElementById('btn-analysis');
const detectorPage = document.getElementById('detector-page');
const analysisPage = document.getElementById('analysis-page');
const tabBtns = document.querySelectorAll('.tab-btn');

function showPage(pageToShow) {
    detectorPage.classList.add('hidden');
    analysisPage.classList.add('hidden');
    pageToShow.classList.remove('hidden');
}

tabBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        tabBtns.forEach(b => {
            b.classList.remove('tab-active');
            b.classList.add('text-gray-400');
        });
        const clickedButton = e.currentTarget;
        clickedButton.classList.add('tab-active');
        clickedButton.classList.remove('text-gray-400');
    });
});

btnDetector.addEventListener('click', () => showPage(detectorPage));

let chartsInitialized = false;
btnAnalysis.addEventListener('click', () => {
    showPage(analysisPage);
    if (!chartsInitialized) {
        setTimeout(() => { 
            initializeCharts();
            chartsInitialized = true;
        }, 100);
    }
});

// --- 3D Background ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('bg-canvas'), alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
const particlesGeometry = new THREE.BufferGeometry;
const particlesCnt = 5000;
const posArray = new Float32Array(particlesCnt * 3);
for (let i = 0; i < particlesCnt * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 5;
}
particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMaterial = new THREE.PointsMaterial({ size: 0.005, color: 0x3b82f6 });
const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
scene.add(particlesMesh);
camera.position.z = 2;
document.addEventListener('mousemove', (event) => {
    particlesMesh.rotation.y = event.clientX * 0.0001;
    particlesMesh.rotation.x = event.clientY * 0.0001;
});
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- ONNX Model and Detection Logic ---
const MODEL_PATH = './best1.onnx'; 
const CLASSES = ["FireExtinguisher", "ToolBox", "OxygenTank"];
const CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.5;
const fileInput = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const outputContainer = document.getElementById('output-container');
const outputCanvas = document.getElementById('output-canvas');
const loader = document.getElementById('loader');
const ctx = outputCanvas.getContext('2d');
let session;
async function loadModel() {
    try {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('Model loaded successfully.');
    } catch (e) {
        console.error('Failed to load the model:', e);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'p-4 mb-4 bg-red-900/50 border border-red-700 text-red-300 rounded-lg text-left';
        errorDiv.innerHTML = `<strong>Error:</strong> Could not load the ONNX model. <br>Please ensure the model file exists at: <code>${MODEL_PATH}</code><br><br>Details: ${e.message}`;
        detectorPage.prepend(errorDiv);
    }
}
loadModel();
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});
function handleFile(file) {
    if (file && session) {
        runObjectDetection(file);
    } else if (!session) {
        console.warn('Model is not loaded yet. Cannot process file.');
    }
}
async function runObjectDetection(file) {
    outputContainer.classList.add('hidden');
    loader.classList.remove('hidden');
    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = async () => {
        try {
            const [input, xRatio, yRatio] = await preprocess(image);
            const feeds = { images: input };
            const results = await session.run(feeds);
            const output = results.output0.data;
            const boxes = postprocess(output, xRatio, yRatio);
            draw(image, boxes);
            outputContainer.classList.remove('hidden');
        } catch (e) {
            console.error('Error during object detection:', e);
        } finally {
            loader.classList.add('hidden');
        }
    };
}
async function preprocess(img) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const modelWidth = 640; const modelHeight = 640;
    canvas.width = modelWidth; canvas.height = modelHeight;
    const xRatio = modelWidth / img.width; const yRatio = modelHeight / img.height;
    context.drawImage(img, 0, 0, modelWidth, modelHeight);
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight);
    const data = imageData.data;
    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255.0);
        green.push(data[i + 1] / 255.0);
        blue.push(data[i + 2] / 255.0);
    }
    const transposedData = red.concat(green, blue);
    const float32Data = new Float32Array(transposedData);
    return [new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]), xRatio, yRatio];
}
function postprocess(output, xRatio, yRatio) {
    const transposed = [];
    const numOutputChannels = CLASSES.length + 4; 
    const numDetections = output.length / numOutputChannels;
    for (let i = 0; i < numDetections; i++) {
        const row = [];
        for (let j = 0; j < numOutputChannels; j++) {
            row.push(output[i + j * numDetections]);
        }
        transposed.push(row);
    }
    const boxes = [];
    for (const row of transposed) {
        const [x, y, w, h, ...classScores] = row;
        let maxScore = 0, classId = -1;
        for(let i=0; i < classScores.length; i++) {
            if (classScores[i] > maxScore) {
                maxScore = classScores[i];
                classId = i;
            }
        }
        if (maxScore > CONFIDENCE_THRESHOLD) {
            boxes.push({
                box: [(x - w / 2) / xRatio, (y - h / 2) / yRatio, w / xRatio, h / yRatio],
                score: maxScore, classId: classId
            });
        }
    }
    return nms(boxes, IOU_THRESHOLD);
}
function nms(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0], box) < iouThreshold);
    }
    return result;
}
function iou(boxA, boxB) {
    const [ax1, ay1, aw, ah] = boxA.box; const [bx1, by1, bw, bh] = boxB.box;
    const ax2 = ax1 + aw, ay2 = ay1 + ah, bx2 = bx1 + bw, by2 = by1 + bh;
    const x_left = Math.max(ax1, bx1), y_top = Math.max(ay1, by1);
    const x_right = Math.min(ax2, bx2), y_bottom = Math.min(ay2, by2);
    if (x_right < x_left || y_bottom < y_top) return 0.0;
    const intersectionArea = (x_right - x_left) * (y_bottom - y_top);
    const boxAArea = aw * ah, boxBArea = bw * bh;
    return intersectionArea / (boxAArea + boxBArea - intersectionArea);
}

function draw(img, boxes) {
    outputCanvas.width = img.width;
    outputCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    boxes.forEach(({ box, score, classId }) => {
        const [x, y, w, h] = box;
        const label = `${CLASSES[classId]} (${(score * 100).toFixed(1)}%)`;
        ctx.strokeStyle = '#0ea5e9';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#0ea5e9';
        ctx.font = 'bold 20px Inter';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 28; 
        ctx.fillRect(x, y - textHeight, textWidth + 10, textHeight);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 5, y - 8);
    });
}

function initializeCharts() {
    const chartTextColor = '#9ca3af';
    const chartGridColor = 'rgba(255, 255, 255, 0.1)';

    const cmLabels = ['Fire Extinguisher', 'ToolBox', 'Oxygen Tank', 'background'];
    const cmData = [
        {x: 0, y: 0, v: 0.94}, {x: 1, y: 0, v: 0.02}, 
        {x: 1, y: 1, v: 0.87}, {x: 3, y: 1, v: 0.33},
        {x: 2, y: 2, v: 0.90}, {x: 3, y: 2, v: 0.67},
        {x: 0, y: 3, v: 0.06}, {x: 1, y: 3, v: 0.12}, {x: 2, y: 3, v: 0.10}
    ];
    new Chart(document.getElementById('confusionMatrixChart'), {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Confidence',
                data: cmData,
                backgroundColor: (ctx) => `rgba(59, 130, 246, ${ctx.raw.v})`,
                borderColor: '#3b82f6',
                borderWidth: 1,
                radius: (ctx) => ctx.raw.v * 25 + 5,
            }]
        },
        options: {
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `True: ${cmLabels[ctx.raw.x]}, Predicted: ${cmLabels[ctx.raw.y]}, Value: ${ctx.raw.v.toFixed(2)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: cmLabels,
                    title: { display: true, text: 'True Class', color: chartTextColor },
                    ticks: { color: chartTextColor },
                    grid: { color: chartGridColor }
                },
                y: {
                    type: 'category',
                    labels: cmLabels.slice().reverse(),
                    offset: true,
                    position: 'left',
                    title: { display: true, text: 'Predicted Class', color: chartTextColor },
                    ticks: { color: chartTextColor, callback: (val, index) => cmLabels[cmLabels.length - 1 - index] },
                    grid: { color: chartGridColor }
                }
            }
        }
    });

    const epochLabels = Array.from({length: 11}, (_, i) => i * 5);
    const lineChartData = {
        'chart1': { title: 'train/box_loss', color: '#2dd4bf', data: [1.02, 0.76, 0.65, 0.58, 0.52, 0.49, 0.47, 0.45, 0.43, 0.42, 0.41] },
        'chart2': { title: 'train/cls_loss', color: '#2dd4bf', data: [2.2, 1.4, 1.0, 0.8, 0.65, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38] },
        'chart3': { title: 'train/dfl_loss', color: '#2dd4bf', data: [1.15, 1.0, 0.95, 0.9, 0.88, 0.85, 0.83, 0.81, 0.8, 0.79, 0.79] },
        'chart4': { title: 'metrics/precision(B)', color: '#38bdf8', data: [0.21, 0.7, 0.82, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96] },
        'chart5': { title: 'metrics/recall(B)', color: '#38bdf8', data: [0.6, 0.75, 0.8, 0.82, 0.85, 0.86, 0.88, 0.89, 0.9, 0.91, 0.91] },
        'chart6': { title: 'val/box_loss', color: '#f471b5', data: [1.1, 0.8, 0.7, 0.65, 0.6, 0.58, 0.55, 0.53, 0.52, 0.51, 0.5] },
        'chart7': { title: 'metrics/mAP50(B)', color: '#a78bfa', data: [0.4, 0.75, 0.85, 0.9, 0.92, 0.93, 0.94, 0.94, 0.94, 0.95, 0.95] },
        'chart8': { title: 'metrics/mAP50-95(B)', color: '#a78bfa', data: [0.4, 0.55, 0.62, 0.65, 0.68, 0.7, 0.71, 0.72, 0.73, 0.73, 0.74] }
    };

    for (const [canvasId, config] of Object.entries(lineChartData)) {
        new Chart(document.getElementById(canvasId), {
            type: 'line',
            data: {
                labels: epochLabels,
                datasets: [{
                    label: config.title,
                    data: config.data,
                    borderColor: config.color,
                    backgroundColor: `${config.color}33`,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                plugins: { legend: { display: false }, title: { display: true, text: config.title, color: chartTextColor } },
                scales: {
                    y: { ticks: { color: chartTextColor }, grid: { color: chartGridColor } },
                    x: { ticks: { color: chartTextColor }, grid: { color: chartGridColor }, title: { display: true, text: 'Epochs', color: chartTextColor } }
                }
            }
        });
    }
}