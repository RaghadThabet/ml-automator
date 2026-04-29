/* ─── State ─── */
let uploadedFile = null;
let selectedTask = null;
let columns      = [];
let trainingDone = false;

const TASK_ALGOS = {
  classification: 'Random Forest + Logistic Regression',
  regression:     'Random Forest Regressor + Linear Regression',
  clustering:     'K-Means + DBSCAN',
};

/* ─── Helpers ─── */
function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function setStep(n) {
  [1, 2, 3, 4].forEach(i => {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
    if (i < n)       el.classList.add('done');
    else if (i === n) el.classList.add('active');
  });
}

/* ─── Slider ─── */
document.getElementById('split-slider').addEventListener('input', function () {
  const v = parseInt(this.value);
  document.getElementById('split-label').textContent = v + '% / ' + (100 - v) + '%';
});

/* ─── Drag & Drop ─── */
const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

/* ─── File handling ─── */
function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['csv', 'xlsx', 'xls'].includes(ext)) {
    alert('Please upload a .csv or .xlsx file.');
    return;
  }
  uploadedFile = file;

  const sizeStr = file.size < 1024 * 1024
    ? (file.size / 1024).toFixed(1) + ' KB'
    : (file.size / 1024 / 1024).toFixed(1) + ' MB';

  document.getElementById('file-info-block').innerHTML = `
    <div class="file-info">
      <div class="file-icon">${ext.toUpperCase()}</div>
      <div>
        <div class="file-name">${file.name}</div>
        <div class="file-meta">${sizeStr}</div>
      </div>
      <button class="file-remove" onclick="removeFile()" title="Remove">×</button>
    </div>`;
  show('file-info-block');

  /*
   * TODO: Replace mock preview with real API call:
   *
   * const form = new FormData();
   * form.append('file', file);
   * fetch('/api/upload', { method: 'POST', body: form })
   *   .then(r => r.json())
   *   .then(data => {
   *     columns = data.columns;
   *     showPreview(data.columns, data.preview_rows);
   *     populateTargetSelect(data.columns);
   *   });
   */
  if (ext === 'csv') {
    const reader = new FileReader();
    reader.onload = e => parseCsvPreview(e.target.result);
    reader.readAsText(file);
  } else {
    mockXlsxPreview();
  }

  setStep(2);
}

function removeFile() {
  uploadedFile = null;
  columns = [];
  document.getElementById('file-info-block').innerHTML = '';
  hide('file-info-block');
  hide('preview-block');
  document.getElementById('file-input').value = '';
  setStep(1);
}

function parseCsvPreview(text) {
  const lines   = text.trim().split('\n').slice(0, 6);
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const rows    = lines.slice(1).map(l => l.split(',').map(c => c.trim().replace(/"/g, '')));
  columns = headers;
  showPreview(headers, rows);
  populateTargetSelect(headers);
}

function mockXlsxPreview() {
  // Remove this function once the real API is connected
  columns = ['age', 'income', 'education', 'job_type', 'loan_status'];
  const rows = [
    ['34', '52000', 'Bachelor',    'Engineer', 'Approved'],
    ['27', '31000', 'High School', 'Sales',    'Denied'],
    ['45', '88000', 'Master',      'Manager',  'Approved'],
    ['52', '61000', 'Bachelor',    'Teacher',  'Approved'],
    ['23', '24000', 'High School', 'Intern',   'Denied'],
  ];
  showPreview(columns, rows);
  populateTargetSelect(columns);
}

function showPreview(headers, rows) {
  const table = document.getElementById('preview-table');
  table.innerHTML =
    '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>' +
    '<tbody>' + rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
  show('preview-block');
}

function populateTargetSelect(cols) {
  const sel = document.getElementById('target-select');
  sel.innerHTML = '<option value="">Select a column…</option>' +
    cols.map(c => `<option value="${c}">${c}</option>`).join('');
}

/* ─── Task selection ─── */
function selectTask(task) {
  selectedTask = task;
  document.querySelectorAll('.task-option').forEach(el => el.classList.remove('selected'));
  document.querySelector(`[data-task="${task}"]`).classList.add('selected');

  task === 'clustering' ? hide('target-field') : show('target-field');

  document.getElementById('algo-display').textContent = TASK_ALGOS[task];

  const dot = document.getElementById('resample-dot');
  const val = document.getElementById('resample-value');
  if (task === 'classification') {
    dot.classList.add('on');
    val.textContent = 'Auto-detect (SMOTE)';
  } else {
    dot.classList.remove('on');
    val.textContent = 'N/A';
  }

  setStep(3);
}

/* ─── Training ─── */
function startTraining() {
  if (!uploadedFile) { alert('Please upload a dataset first.'); return; }
  if (!selectedTask) { alert('Please select an ML task.'); return; }
  if (selectedTask !== 'clustering' && !document.getElementById('target-select').value) {
    alert('Please select a target column.'); return;
  }

  document.getElementById('run-btn').disabled = true;
  show('card-progress');
  document.getElementById('card-progress').scrollIntoView({ behavior: 'smooth', block: 'start' });

  /*
   * TODO: Replace simulation with real API call:
   *
   * const payload = {
   *   task:   selectedTask,
   *   target: document.getElementById('target-select').value,
   *   split:  parseInt(document.getElementById('split-slider').value) / 100,
   * };
   * fetch('/api/train', {
   *   method: 'POST',
   *   headers: { 'Content-Type': 'application/json' },
   *   body: JSON.stringify(payload),
   * })
   *   .then(r => r.json())
   *   .then(data => renderResults(data));
   */

  const splitLabel = document.getElementById('split-label').textContent;
  const algo1 = TASK_ALGOS[selectedTask].split(' + ')[0];
  const algo2 = TASK_ALGOS[selectedTask].split(' + ')[1];

  const logs = [
    { t: '00:01', msg: 'Uploading dataset…',                                          cls: 'info' },
    { t: '00:02', msg: `File accepted — 1,240 rows × ${columns.length || 5} columns`, cls: 'ok'   },
    { t: '00:03', msg: 'Handling missing values (median/mode imputation)…',            cls: 'info' },
    { t: '00:04', msg: 'Missing values filled — 0 nulls remaining',                   cls: 'ok'   },
    { t: '00:05', msg: 'Encoding categorical variables (one-hot)…',                   cls: 'info' },
    { t: '00:06', msg: 'Encoding done — feature matrix: 1240 × 18',                   cls: 'ok'   },
    { t: '00:07', msg: 'Scaling numerical features (StandardScaler)…',                cls: 'info' },
    { t: '00:08', msg: 'Scaling complete',                                             cls: 'ok'   },
    { t: '00:09', msg: `Splitting data ${splitLabel}…`,                                cls: 'info' },
    { t: '00:10', msg: 'Train set: 992 — Test set: 248',                               cls: 'ok'   },
    { t: '00:12', msg: `Training model 1: ${algo1}…`,                                 cls: 'info' },
    { t: '00:18', msg: 'Model 1 done',                                                 cls: 'ok'   },
    { t: '00:19', msg: `Training model 2: ${algo2}…`,                                 cls: 'info' },
    { t: '00:23', msg: 'Model 2 done',                                                 cls: 'ok'   },
    { t: '00:24', msg: 'Evaluating both models on test set…',                         cls: 'info' },
    { t: '00:25', msg: 'Best model selected — pipeline complete ✓',                   cls: 'ok'   },
  ];

  const bar    = document.getElementById('progress-bar');
  const logOut = document.getElementById('log-output');
  let i = 0;

  const interval = setInterval(() => {
    if (i >= logs.length) {
      clearInterval(interval);
      bar.style.width = '100%';
      setTimeout(showResults, 400);
      return;
    }
    const l = logs[i];
    bar.style.width = Math.round((i + 1) / logs.length * 100) + '%';
    const div = document.createElement('div');
    div.className = 'log-line';
    div.innerHTML = `<span class="ts">${l.t}</span><span class="${l.cls}">${l.msg}</span>`;
    logOut.appendChild(div);
    logOut.scrollTop = logOut.scrollHeight;
    i++;
  }, 260);
}

/* ─── Results ─── */
function showResults() {
  setStep(4);
  show('card-results');
  document.getElementById('card-results').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const algo1 = TASK_ALGOS[selectedTask].split(' + ')[0];
  const algo2 = TASK_ALGOS[selectedTask].split(' + ')[1];

  let score1, score2, metric;
  if (selectedTask === 'classification') { score1 = '0.924'; score2 = '0.871'; metric = 'Accuracy';   }
  else if (selectedTask === 'regression'){ score1 = '0.912'; score2 = '0.843'; metric = 'R²';         }
  else                                   { score1 = '0.681'; score2 = '0.594'; metric = 'Silhouette'; }

  document.getElementById('algo-compare-section').innerHTML = `
    <div class="algo-card winner">
      <div class="winner-badge">Selected</div>
      <div class="algo-name">${algo1}</div>
      <div class="algo-score">${metric}: ${score1}</div>
    </div>
    <div class="algo-card">
      <div class="algo-name">${algo2}</div>
      <div class="algo-score">${metric}: ${score2}</div>
    </div>`;

  document.getElementById('task-badge').textContent = selectedTask;
  buildMetrics();
  buildVisual();
  buildFeatureImportance();
  trainingDone = true;
}

function buildMetrics() {
  let cards = [];
  if (selectedTask === 'classification') {
    cards = [
      { label: 'Accuracy',  value: '92.4%' },
      { label: 'Precision', value: '91.8%' },
      { label: 'Recall',    value: '93.1%' },
      { label: 'F1-Score',  value: '92.4%' },
    ];
  } else if (selectedTask === 'regression') {
    cards = [
      { label: 'R² Score', value: '0.912' },
      { label: 'MAE',      value: '1,240' },
      { label: 'MSE',      value: '3.1M'  },
    ];
  } else {
    cards = [
      { label: 'Silhouette Score', value: '0.681'  },
      { label: 'Clusters found',   value: '4'      },
      { label: 'Inertia',          value: '12,840' },
    ];
  }
  document.getElementById('metrics-grid').innerHTML =
    cards.map(c => `
      <div class="metric-card">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}</div>
      </div>`).join('');
}

function buildVisual() {
  const title   = document.getElementById('visual-title');
  const content = document.getElementById('visual-content');

  if (selectedTask === 'classification') {
    title.textContent = 'Confusion matrix';
    content.innerHTML = `
      <p style="font-size:11px;color:var(--text-tertiary);margin-bottom:6px">Predicted →</p>
      <div class="confusion-grid">
        <div class="cm-cell cm-label" style="font-size:10px">Actual →</div>
        <div class="cm-cell cm-label">Positive</div>
        <div class="cm-cell cm-label">Negative</div>
        <div class="cm-cell cm-label">Positive</div>
        <div class="cm-cell cm-tp">112<div class="cm-sub">TP</div></div>
        <div class="cm-cell cm-fn">9<div class="cm-sub">FN</div></div>
        <div class="cm-cell cm-label">Negative</div>
        <div class="cm-cell cm-fp">10<div class="cm-sub">FP</div></div>
        <div class="cm-cell cm-tn">117<div class="cm-sub">TN</div></div>
      </div>`;
  } else if (selectedTask === 'regression') {
    title.textContent = 'Residual plot';
    content.innerHTML = `
      <p style="font-size:13px;color:var(--text-secondary);margin-top:8px">
        Residual plot renders here via Chart.js or Plotly.
      </p>
      <div class="api-note">GET /api/results/plot → returns chart data or base64 image</div>`;
  } else {
    title.textContent = 'Cluster visualization';
    content.innerHTML = `
      <p style="font-size:13px;color:var(--text-secondary);margin-top:8px">
        2D scatter (PCA-reduced) renders here.
      </p>
      <div class="api-note">GET /api/results/clusters → returns cluster assignments + PCA coords</div>`;
  }
}

function buildFeatureImportance() {
  const feats = columns.length >= 4
    ? columns.slice(0, -1)
    : ['age', 'income', 'education', 'job_type'];
  const importances = [0.38, 0.27, 0.19, 0.16].slice(0, feats.length);

  document.getElementById('fi-section').innerHTML =
    feats.map((f, i) => `
      <div class="fi-row">
        <div class="fi-feature">${f}</div>
        <div class="fi-bar-bg">
          <div class="fi-bar" style="width:${Math.round(importances[i] * 100)}%"></div>
        </div>
        <div class="fi-pct">${Math.round(importances[i] * 100)}%</div>
      </div>`).join('');
}

/* ─── Save model ─── */
function saveModel() {
  if (!trainingDone) { alert('No trained model yet. Please run training first.'); return; }

  /*
   * TODO: Replace with real download:
   *
   * fetch('/api/model/download', { method: 'POST' })
   *   .then(r => r.blob())
   *   .then(blob => {
   *     const url = URL.createObjectURL(blob);
   *     const a   = document.createElement('a');
   *     a.href     = url;
   *     a.download = 'model.pkl';
   *     a.click();
   *     URL.revokeObjectURL(url);
   *   });
   */

  const btn = document.getElementById('save-btn');
  btn.textContent = 'Preparing download…';
  btn.disabled = true;

  setTimeout(() => {
    btn.disabled = false;
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" stroke-width="1.5" style="width:14px;height:14px;stroke:currentColor;fill:none">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5
             M12 3v13.5m0 0l-4.5-4.5M12 16.5l4.5-4.5"/>
      </svg>
      Save model (.pkl)`;
    alert('Ready! Connect POST /api/model/download to trigger the real file download.');
  }, 1200);
}