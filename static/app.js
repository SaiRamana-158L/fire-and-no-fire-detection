/**
 * UAV Fire & No-Fire Detection Dashboard Frontend Logic
 * LightWeight Deep-Learning & Fused-Feature (FF) Pipeline
 */

// State Management
const state = {
  currentModule: 'module-1',
  currentSample: '/static/dataset/fire/uav_fire_01.jpg',
  activeModel: 'fused_feature',
  clipLimit: 2.0,
  tileSize: 8,
  showHeatmap: true,
  audioAlarmEnabled: false,
  alarmInterval: null,
  audioCtx: null,
  samples: { fire: [], no_fire: [] },
  metricsData: null,
  charts: {
    histogram: null,
    models: null
  },
  hudTelemetry: {
    lat: 44.3824,
    lon: -121.5492,
    alt: 185,
    spd: 14.2,
    bat: 88,
    heading: 0
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  setupNavigation();
  setupKaggleModal();
  setupCLAHEControls();
  setupFileUpload();
  setupHUDControls();
  setupTelemetryLoop();

  // Load Data from FastAPI Backend
  await loadMetrics();
  await loadSampleGallery();

  // Run initial CLAHE processing & live prediction on default sample
  triggerCLAHE();
  runPrediction();
});

// ================= NAVIGATION ================= //
function setupNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetModule = btn.getAttribute('data-module');
      switchModule(targetModule);
    });
  });
}

function switchModule(moduleId) {
  state.currentModule = moduleId;

  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-module') === moduleId);
  });

  document.querySelectorAll('.module-section').forEach(sec => {
    sec.classList.toggle('active', sec.id === moduleId);
  });

  // Re-render chart if switching to module 2
  if (moduleId === 'module-2' && state.metricsData) {
    setTimeout(renderModelsChart, 50);
  }
}

// ================= KAGGLE MODAL ================= //
function setupKaggleModal() {
  const modal = document.getElementById('kaggle-modal');
  const btnShow = document.getElementById('btn-show-kaggle-guide');
  const btnClose = document.getElementById('btn-close-modal');
  const btnOk = document.getElementById('btn-modal-ok');

  btnShow?.addEventListener('click', () => modal.classList.add('active'));
  btnClose?.addEventListener('click', () => modal.classList.remove('active'));
  btnOk?.addEventListener('click', () => modal.classList.remove('active'));
  modal?.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('active');
  });
}

// ================= DATA LOADING ================= //
async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Failed to load metrics');
    const data = await res.json();
    state.metricsData = data;
    renderMetricsTable(data.models);
    renderModelsChart('bar');
  } catch (err) {
    console.error('Error loading metrics:', err);
  }
}

async function loadSampleGallery() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) throw new Error('Failed to load samples');
    const data = await res.json();
    state.samples = {
      fire: data.fire || [],
      no_fire: data.no_fire || [],
      uploaded: []
    };
    renderSampleGrid('all');
    renderQuickChips();
  } catch (err) {
    console.error('Error loading sample gallery:', err);
  }
}

// ================= MODULE 1: DATASET & CLAHE ================= //
function renderSampleGrid(filter) {
  const grid = document.getElementById('sample-grid');
  if (!grid) return;
  grid.innerHTML = '';

  let list = [];
  // Show uploaded images first!
  if (state.samples.uploaded && state.samples.uploaded.length > 0) {
    state.samples.uploaded.forEach(u => {
      list.push({ src: u.src, type: 'uploaded', label: u.label || 'Uploaded' });
    });
  }

  if (filter === 'all' || filter === 'fire') {
    state.samples.fire.forEach(s => list.push({ src: s, type: 'fire', label: 'Fire' }));
  }
  if (filter === 'all' || filter === 'no_fire') {
    state.samples.no_fire.forEach(s => list.push({ src: s, type: 'nofire', label: 'No-Fire' }));
  }

  list.forEach(item => {
    const div = document.createElement('div');
    const isSelected = item.src === state.currentSample;
    div.className = `sample-thumb ${isSelected ? 'selected' : ''}`;
    div.innerHTML = `
      <img src="${item.src}" alt="${item.label}">
      <span class="thumb-tag ${item.type === 'uploaded' ? 'badge-success' : item.type}">${item.label}</span>
    `;
    div.addEventListener('click', () => {
      document.querySelectorAll('.sample-thumb').forEach(t => t.classList.remove('selected'));
      div.classList.add('selected');
      state.currentSample = item.src;
      triggerCLAHE();
      runPrediction();
    });
    grid.appendChild(div);
  });

  // Filter Buttons
  document.querySelectorAll('.sample-filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-filter') === filter);
    btn.onclick = () => renderSampleGrid(btn.getAttribute('data-filter'));
  });
}

function setupCLAHEControls() {
  const sliderClip = document.getElementById('slider-clip-limit');
  const lblClip = document.getElementById('lbl-clip-limit');
  const gridBtns = document.querySelectorAll('.grid-btn');

  sliderClip?.addEventListener('input', (e) => {
    state.clipLimit = parseFloat(e.target.value);
    lblClip.textContent = state.clipLimit.toFixed(1);
    triggerCLAHE();
  });

  gridBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      gridBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.tileSize = parseInt(btn.getAttribute('data-size'));
      document.getElementById('lbl-tile-size').innerHTML = `${state.tileSize}&times;${state.tileSize}`;
      triggerCLAHE();
    });
  });
}

function showUploadStatus(message, type = 'loading') {
  const statusEl = document.getElementById('upload-status');
  if (!statusEl) return;
  statusEl.className = `upload-status ${type}`;
  let iconHtml = '';
  if (type === 'loading') {
    iconHtml = '<i data-lucide="loader-2" class="spin"></i>';
  } else if (type === 'error') {
    iconHtml = '<i data-lucide="alert-circle"></i>';
  } else {
    iconHtml = '<i data-lucide="check-circle-2"></i>';
  }
  statusEl.innerHTML = `${iconHtml} <span>${message}</span>`;
  if (window.lucide) window.lucide.createIcons();

  if (type !== 'loading') {
    setTimeout(() => {
      statusEl.className = 'upload-status hidden';
    }, 5000);
  }
}

function setupFileUpload() {
  const dropZone = document.getElementById('upload-zone');
  const fileInputM1 = document.getElementById('file-input-m1');
  const btnBrowseM1 = document.getElementById('btn-browse-m1');
  const droneHud = document.getElementById('drone-hud');
  const fileInputM3 = document.getElementById('file-input-m3');
  const btnUploadM3 = document.getElementById('btn-upload-m3');

  // Prevent default window drag/drop behavior (prevent browser opening dropped images)
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    window.addEventListener(name, (e) => {
      e.preventDefault();
    }, false);
  });

  // Module 1: Click to Browse
  dropZone?.addEventListener('click', (e) => {
    if (e.target !== fileInputM1) {
      fileInputM1.click();
    }
  });

  btnBrowseM1?.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInputM1.click();
  });

  fileInputM1?.addEventListener('click', (e) => {
    e.stopPropagation();
  });

  fileInputM1?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCustomFile(e.target.files[0]);
      e.target.value = ''; // Reset so the same file can be re-selected
    }
  });

  // Module 1: Drag and Drop Handlers
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-active');
      }, false);
    });

    ['dragleave', 'dragend'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-active');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-active');

      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleCustomFile(dt.files[0]);
      }
    }, false);
  }

  // Module 3: Upload button & Drone HUD Drag & Drop
  btnUploadM3?.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInputM3?.click();
  });

  fileInputM3?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCustomFile(e.target.files[0]);
      e.target.value = '';
    }
  });

  if (droneHud) {
    ['dragenter', 'dragover'].forEach(name => {
      droneHud.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        droneHud.classList.add('hud-drag-active');
      }, false);
    });

    ['dragleave', 'dragend', 'drop'].forEach(name => {
      droneHud.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        droneHud.classList.remove('hud-drag-active');
      }, false);
    });

    droneHud.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      droneHud.classList.remove('hud-drag-active');
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleCustomFile(dt.files[0]);
      }
    }, false);
  }
}

async function handleCustomFile(file) {
  if (!file) return;

  const fileName = file.name || 'uploaded_uav.jpg';
  const fileSizeKb = (file.size / 1024).toFixed(1);
  showUploadStatus(`Reading & processing ${fileName} (${fileSizeKb} KB)...`, 'loading');

  // Read file via FileReader
  const reader = new FileReader();
  reader.onerror = () => {
    showUploadStatus(`Failed to read file ${fileName}`, 'error');
  };

  reader.onload = async (e) => {
    const dataUrl = e.target.result;
    const imgRaw = document.getElementById('img-raw');
    const hudImg = document.getElementById('drone-live-feed');
    if (imgRaw) imgRaw.src = dataUrl;
    if (hudImg) hudImg.src = dataUrl;

    try {
      const payload = {
        image_base64: dataUrl,
        clip_limit: state.clipLimit,
        tile_size: state.tileSize
      };

      const res = await fetch('/api/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `Server responded with ${res.status}`);
      }
      const data = await res.json();

      if (data.saved_url) {
        state.currentSample = data.saved_url;

        // Add to uploaded samples list
        if (!state.samples.uploaded) state.samples.uploaded = [];
        const exists = state.samples.uploaded.some(u => u.src === data.saved_url);
        if (!exists) {
          state.samples.uploaded.unshift({
            src: data.saved_url,
            label: fileName.length > 14 ? fileName.substring(0, 12) + '...' : fileName
          });
        }

        renderSampleGrid('all');
        renderQuickChips();
      }

      updateCLAHEVisuals(data);

      // Predict on the newly saved image sample_path directly via JSON
      const predPayload = {
        sample_path: data.saved_url || state.currentSample,
        image_base64: dataUrl,
        model_key: state.activeModel,
        clip_limit: state.clipLimit,
        tile_size: state.tileSize
      };

      const pRes = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predPayload)
      });

      if (!pRes.ok) throw new Error('Prediction failed');
      const pData = await pRes.json();
      state.lastPrediction = pData;
      updatePredictionVisuals(pData);

      showUploadStatus(`✓ Processed & Classified: ${fileName} (${data.original_resolution})`, 'success');
    } catch (err) {
      console.error('Upload processing error:', err);
      showUploadStatus(`Error: ${err.message}`, 'error');
    }
  };

  reader.readAsDataURL(file);
}

async function triggerCLAHE() {
  const formData = new FormData();
  formData.append('sample_path', state.currentSample);
  formData.append('clip_limit', state.clipLimit);
  formData.append('tile_size', state.tileSize);

  try {
    const res = await fetch('/api/preprocess', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('CLAHE API error');
    const data = await res.json();
    updateCLAHEVisuals(data);
  } catch (err) {
    console.error('Error running CLAHE:', err);
  }
}

function updateCLAHEVisuals(data) {
  const imgRaw = document.getElementById('img-raw');
  const imgClahe = document.getElementById('img-clahe');
  const imgResized = document.getElementById('img-resized');
  const lblRes = document.getElementById('lbl-raw-res');

  if (imgRaw && data.raw_base64) imgRaw.src = data.raw_base64;
  if (imgClahe && data.clahe_base64) imgClahe.src = data.clahe_base64;
  if (imgResized && data.resized_base64) imgResized.src = data.resized_base64;
  if (lblRes) lblRes.textContent = data.original_resolution;

  renderHistogramChart(data.hist_raw, data.hist_clahe);
}

function renderHistogramChart(histRaw, histClahe) {
  const ctx = document.getElementById('chart-histogram')?.getContext('2d');
  if (!ctx) return;

  const labels = Array.from({ length: 32 }, (_, i) => i * 8);

  if (state.charts.histogram) {
    state.charts.histogram.destroy();
  }

  state.charts.histogram = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Raw Luminance (Haze/Uncalibrated)',
          data: histRaw,
          borderColor: 'rgba(156, 163, 175, 0.6)',
          backgroundColor: 'rgba(156, 163, 175, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          label: 'CLAHE Equalized (Contrast Amplified)',
          data: histClahe,
          borderColor: '#ff4500',
          backgroundColor: 'rgba(255, 69, 0, 0.15)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#9ca3af', font: { family: 'Outfit', size: 10 } }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#6b7280', font: { size: 9 } }
        },
        y: {
          display: false
        }
      }
    }
  });
}

// ================= MODULE 2: LIGHTWEIGHT MODELS ================= //
function renderMetricsTable(models) {
  const tbody = document.getElementById('metrics-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const order = ['mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3', 'nasnet_mobile', 'fused_feature'];

  order.forEach(key => {
    const m = models[key];
    if (!m) return;
    const isBest = m.is_best;
    const tr = document.createElement('tr');
    if (isBest) tr.className = 'highlight-row';

    tr.innerHTML = `
      <td>
        <strong>${m.name}</strong>
        ${isBest ? '<span class="best-badge">BEST (94.66%)</span>' : ''}
      </td>
      <td><code>${m.params}</code></td>
      <td><code>${m.latency_ms} ms</code></td>
      <td><strong style="color: ${isBest ? '#a78bfa' : '#34d399'}">${m.accuracy.toFixed(2)}%</strong></td>
      <td>${m.precision.toFixed(2)}%</td>
      <td>${m.sensitivity.toFixed(2)}%</td>
      <td>${m.f1_score.toFixed(2)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderModelsChart(type = 'bar') {
  const ctx = document.getElementById('chart-models')?.getContext('2d');
  if (!ctx || !state.metricsData) return;

  const models = state.metricsData.models;
  const labels = ['MobileNetV1', 'MobileNetV2', 'MobileNetV3', 'NASNetMobile', 'Proposed FF'];
  const accs = [
    models.mobilenet_v1.accuracy,
    models.mobilenet_v2.accuracy,
    models.mobilenet_v3.accuracy,
    models.nasnet_mobile.accuracy,
    models.fused_feature.accuracy
  ];
  const precs = [
    models.mobilenet_v1.precision,
    models.mobilenet_v2.precision,
    models.mobilenet_v3.precision,
    models.nasnet_mobile.precision,
    models.fused_feature.precision
  ];
  const f1s = [
    models.mobilenet_v1.f1_score,
    models.mobilenet_v2.f1_score,
    models.mobilenet_v3.f1_score,
    models.nasnet_mobile.f1_score,
    models.fused_feature.f1_score
  ];

  if (state.charts.models) {
    state.charts.models.destroy();
  }

  // Handle Chart View Toggles
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-chart') === type);
    btn.onclick = () => renderModelsChart(btn.getAttribute('data-chart'));
  });

  if (type === 'radar') {
    state.charts.models = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score'],
        datasets: [
          {
            label: 'Proposed Fused-Feature (FF)',
            data: [
              models.fused_feature.accuracy,
              models.fused_feature.precision,
              models.fused_feature.sensitivity,
              models.fused_feature.specificity,
              models.fused_feature.f1_score
            ],
            borderColor: '#a78bfa',
            backgroundColor: 'rgba(167, 139, 250, 0.25)',
            borderWidth: 2
          },
          {
            label: 'MobileNetV3-Small',
            data: [
              models.mobilenet_v3.accuracy,
              models.mobilenet_v3.precision,
              models.mobilenet_v3.sensitivity,
              models.mobilenet_v3.specificity,
              models.mobilenet_v3.f1_score
            ],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            borderWidth: 1.5
          },
          {
            label: 'MobileNetV1',
            data: [
              models.mobilenet_v1.accuracy,
              models.mobilenet_v1.precision,
              models.mobilenet_v1.sensitivity,
              models.mobilenet_v1.specificity,
              models.mobilenet_v1.f1_score
            ],
            borderColor: '#9ca3af',
            backgroundColor: 'rgba(156, 163, 175, 0.05)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 85,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            pointLabels: { color: '#9ca3af', font: { family: 'Outfit', size: 10 } },
            ticks: { display: false }
          }
        },
        plugins: {
          legend: { labels: { color: '#d1d5db', font: { family: 'Outfit' } } }
        }
      }
    });
  } else {
    state.charts.models = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accs,
            backgroundColor: [
              'rgba(56, 189, 248, 0.5)',
              'rgba(56, 189, 248, 0.5)',
              'rgba(56, 189, 248, 0.65)',
              'rgba(56, 189, 248, 0.5)',
              'rgba(255, 69, 0, 0.85)' // Fused feature gets fire accent
            ],
            borderColor: [
              '#38bdf8',
              '#38bdf8',
              '#38bdf8',
              '#38bdf8',
              '#ff4500'
            ],
            borderWidth: 1.5,
            borderRadius: 6
          },
          {
            label: 'F1-Score (%)',
            data: f1s,
            backgroundColor: 'rgba(139, 92, 246, 0.35)',
            borderColor: '#a78bfa',
            borderWidth: 1.5,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#d1d5db', font: { family: 'Outfit' } } }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 10 } }
          },
          y: {
            min: 85,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 9 } }
          }
        }
      }
    });
  }
}

function updateVectorVisualizer(rawVec, fusedVec) {
  const rawContainer = document.getElementById('raw-vector-cells');
  const fusedContainer = document.getElementById('fused-vector-cells');
  if (!rawContainer || !fusedContainer) return;

  rawContainer.innerHTML = '';
  fusedContainer.innerHTML = '';

  rawVec.forEach(v => {
    const span = document.createElement('span');
    span.className = 'v-cell';
    span.textContent = v.toFixed(2);
    rawContainer.appendChild(span);
  });

  fusedVec.forEach(v => {
    const span = document.createElement('span');
    span.className = 'v-cell';
    span.textContent = v.toFixed(2);
    fusedContainer.appendChild(span);
  });
}

// ================= MODULE 3: LIVE SURVEILLANCE & PREDICTION ================= //
function setupHUDControls() {
  const selectModel = document.getElementById('select-inference-model');
  const btnNext = document.getElementById('btn-toggle-cam');
  const btnHeatmap = document.getElementById('btn-toggle-heatmap');
  const btnAlarm = document.getElementById('btn-toggle-alarm');

  selectModel?.addEventListener('change', (e) => {
    state.activeModel = e.target.value;
    runPrediction();
  });

  btnNext?.addEventListener('click', () => {
    cycleNextSample();
  });

  btnHeatmap?.addEventListener('click', () => {
    state.showHeatmap = !state.showHeatmap;
    btnHeatmap.classList.toggle('active', state.showHeatmap);
    const hudImg = document.getElementById('drone-live-feed');
    if (hudImg && state.lastPrediction) {
      hudImg.src = state.showHeatmap ? state.lastPrediction.gradcam_base64 : state.lastPrediction.processed_base64;
    }
  });

  btnAlarm?.addEventListener('click', () => {
    toggleAudioAlarm();
  });
}

function renderQuickChips() {
  const container = document.getElementById('quick-test-chips');
  if (!container) return;
  container.innerHTML = '';

  const all = [
    ...state.samples.fire.map(s => ({ src: s, name: 'Fire Sample' })),
    ...state.samples.no_fire.map(s => ({ src: s, name: 'No-Fire Sample' }))
  ];

  all.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.className = `chip-btn ${item.src === state.currentSample ? 'active' : ''}`;
    btn.textContent = `${item.name} #${idx + 1}`;
    btn.onclick = () => {
      document.querySelectorAll('.chip-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentSample = item.src;
      triggerCLAHE();
      runPrediction();
    };
    container.appendChild(btn);
  });
}

function cycleNextSample() {
  const all = [...state.samples.fire, ...state.samples.no_fire];
  if (all.length === 0) return;
  const currentIdx = all.indexOf(state.currentSample);
  const nextIdx = (currentIdx + 1) % all.length;
  state.currentSample = all[nextIdx];

  // Update selection everywhere
  renderSampleGrid('all');
  renderQuickChips();
  triggerCLAHE();
  runPrediction();
}

async function runPrediction() {
  const formData = new FormData();
  formData.append('sample_path', state.currentSample);
  formData.append('model_key', state.activeModel);
  formData.append('clip_limit', state.clipLimit);
  formData.append('tile_size', state.tileSize);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Prediction API error');
    const data = await res.json();
    state.lastPrediction = data;
    updatePredictionVisuals(data);
  } catch (err) {
    console.error('Prediction failed:', err);
  }
}

function updatePredictionVisuals(data) {
  const pred = data.prediction;
  const isFire = pred.label === 'FIRE DETECTED';

  // Live HUD image
  const hudImg = document.getElementById('drone-live-feed');
  if (hudImg) {
    hudImg.src = state.showHeatmap ? data.gradcam_base64 : data.processed_base64;
  }

  // Previews
  const imgGradCam = document.getElementById('img-gradcam-preview');
  const imgEvalClahe = document.getElementById('img-eval-clahe');
  if (imgGradCam) imgGradCam.src = data.gradcam_base64;
  if (imgEvalClahe) imgEvalClahe.src = data.processed_base64;

  // Banner & Alert
  const banner = document.getElementById('fire-alert-banner');
  const bannerTitle = document.getElementById('alert-banner-title');
  const bannerDesc = document.getElementById('alert-banner-desc');
  const hudStatus = document.getElementById('hud-status-badge');

  if (isFire) {
    banner?.classList.remove('clear-state');
    if (bannerTitle) bannerTitle.textContent = `${pred.alert_level} FOREST FIRE DETECTED`;
    if (bannerDesc) bannerDesc.innerHTML = `Drone Telemetry: ${state.hudTelemetry.lat.toFixed(4)}&deg; N, ${Math.abs(state.hudTelemetry.lon).toFixed(4)}&deg; W &bull; Confidence: <strong>${pred.confidence_fire}% Fire</strong> &bull; Flames confirmed via ${data.model_info.name}.`;
    if (hudStatus) {
      hudStatus.className = 'hud-badge-feed';
      hudStatus.innerHTML = `<i data-lucide="flame"></i> FIRE DETECTED (${pred.confidence_fire}%)`;
    }
    // Auto beep if alarm toggled
    if (state.audioAlarmEnabled) triggerBeep();
  } else {
    banner?.classList.add('clear-state');
    if (bannerTitle) bannerTitle.textContent = 'NO FIRE DETECTED &bull; CLEAR FOREST CANOPY';
    if (bannerDesc) bannerDesc.innerHTML = `Drone Telemetry: ${state.hudTelemetry.lat.toFixed(4)}&deg; N, ${Math.abs(state.hudTelemetry.lon).toFixed(4)}&deg; W &bull; Confidence: <strong>${pred.confidence_nofire}% Clear</strong> &bull; No thermal or smoke hazard detected.`;
    if (hudStatus) {
      hudStatus.className = 'hud-badge-feed clear';
      hudStatus.innerHTML = `<i data-lucide="trees"></i> CLEAR CANOPY (${pred.confidence_nofire}%)`;
    }
  }

  if (window.lucide) window.lucide.createIcons();

  // Prediction Result Box
  const pBox = document.getElementById('prediction-result-box');
  const pLabel = document.getElementById('pred-class-label');
  const pSub = document.getElementById('pred-model-sub');
  const pScore = document.getElementById('pred-confidence-score');

  if (pBox) pBox.className = `prediction-box ${isFire ? 'danger' : 'safe'}`;
  if (pLabel) pLabel.textContent = pred.label;
  if (pSub) pSub.textContent = `SoftMax Output via ${data.model_info.name} (50% reduction)`;
  if (pScore) pScore.textContent = `${isFire ? pred.confidence_fire : pred.confidence_nofire}%`;

  // SoftMax Bars
  const lblFire = document.getElementById('lbl-prob-fire');
  const barFire = document.getElementById('bar-prob-fire');
  const lblNoFire = document.getElementById('lbl-prob-nofire');
  const barNoFire = document.getElementById('bar-prob-nofire');

  if (lblFire) lblFire.textContent = `${pred.confidence_fire}%`;
  if (barFire) barFire.style.width = `${pred.confidence_fire}%`;
  if (lblNoFire) lblNoFire.textContent = `${pred.confidence_nofire}%`;
  if (barNoFire) barNoFire.style.width = `${pred.confidence_nofire}%`;

  // Execution Time Badge
  const execBadge = document.getElementById('badge-exec-time');
  const statLatency = document.getElementById('stat-latency');
  if (execBadge) execBadge.textContent = `${data.execution_time_ms} ms Latency`;
  if (statLatency) statLatency.textContent = `${data.execution_time_ms} ms`;

  // Update sample feature vectors
  if (data.feature_vectors) {
    updateVectorVisualizer(data.feature_vectors.extracted_sample, data.feature_vectors.reduced_50pct_sample);
  }
}

// ================= TELEMETRY SIMULATION & AUDIO ================= //
function setupTelemetryLoop() {
  setInterval(() => {
    // Subtle realistic UAV drift
    state.hudTelemetry.lat += (Math.random() - 0.5) * 0.0001;
    state.hudTelemetry.lon += (Math.random() - 0.5) * 0.0001;
    state.hudTelemetry.alt = Math.round(185 + Math.sin(Date.now() / 2000) * 4);
    state.hudTelemetry.spd = (14.2 + (Math.random() - 0.5) * 0.4).toFixed(1);

    const hudLat = document.getElementById('hud-lat');
    const hudLon = document.getElementById('hud-lon');
    const hudAlt = document.getElementById('hud-alt');
    const hudSpd = document.getElementById('hud-spd');

    if (hudLat) hudLat.innerHTML = `${state.hudTelemetry.lat.toFixed(4)}&deg; N`;
    if (hudLon) hudLon.innerHTML = `${Math.abs(state.hudTelemetry.lon).toFixed(4)}&deg; W`;
    if (hudAlt) hudAlt.innerHTML = `${state.hudTelemetry.alt} m AGL`;
    if (hudSpd) hudSpd.innerHTML = `${state.hudTelemetry.spd} m/s`;
  }, 1000);
}

function toggleAudioAlarm() {
  state.audioAlarmEnabled = !state.audioAlarmEnabled;
  const btn = document.getElementById('btn-toggle-alarm');
  if (btn) {
    btn.innerHTML = state.audioAlarmEnabled
      ? '<i data-lucide="volume-x"></i> Silence Alarm'
      : '<i data-lucide="volume-2"></i> Sound Alarm';
    btn.className = state.audioAlarmEnabled ? 'btn btn-primary btn-sm' : 'btn btn-warning btn-sm';
    if (window.lucide) window.lucide.createIcons();
  }

  if (state.audioAlarmEnabled) {
    triggerBeep();
  }
}

function triggerBeep() {
  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, state.audioCtx.currentTime); // A5 note
    osc.frequency.exponentialRampToValueAtTime(440, state.audioCtx.currentTime + 0.25);

    gain.gain.setValueAtTime(0.2, state.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, state.audioCtx.currentTime + 0.25);

    osc.connect(gain);
    gain.connect(state.audioCtx.destination);

    osc.start();
    osc.stop(state.audioCtx.currentTime + 0.26);
  } catch (e) {
    console.warn('Audio not allowed without gesture', e);
  }
}
