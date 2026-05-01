/* ─── State ─── */
let uploadedFile = null;
let selectedTask = null;
let columns      = [];
let trainingDone = false;

const API = 'http://localhost:8000';   // change if backend runs elsewhere

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
    if (i < n)        el.classList.add('done');
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
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
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

  // ── Real API call to /api/upload ──
  const form = new FormData();
  form.append('file', file);

  fetch(`${API}/api/upload`, { method: 'POST', body: form })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'Upload failed'); });
      return r.json();
    })
    .then(data => {
      columns = data.columns;
      showPreview(data.columns, data.preview_rows);
      populateTargetSelect(data.columns);

      // Update file-meta with real row count
      document.querySelector('.file-meta').textContent =
        `${sizeStr}  ·  ${data.rows.toLocaleString()} rows × ${data.columns.length} columns`;
    })
    .catch(err => {
      alert('Upload error: ' + err.message);
      removeFile();
    });

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

function showPreview(headers, rows) {
  const table = document.getElementById('preview-table');
  table.innerHTML =
    '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>' +
    '<tbody>' + rows.map(r =>
      '<tr>' + r.map(c => `<td>${c === null || c === undefined ? '' : c}</td>`).join('') + '</tr>'
    ).join('') + '</tbody>';
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
  if (!uploadedFile)  { alert('Please upload a dataset first.'); return; }
  if (!selectedTask)  { alert('Please select an ML task.'); return; }
  if (selectedTask !== 'clustering' && !document.getElementById('target-select').value) {
    alert('Please select a target column.'); return;
  }

  document.getElementById('run-btn').disabled = true;
  show('card-progress');
  hide('card-results');
  document.getElementById('card-progress').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const bar    = document.getElementById('progress-bar');
  const logOut = document.getElementById('log-output');
  logOut.innerHTML = '';
  bar.style.width = '0%';

  const splitPct = parseInt(document.getElementById('split-slider').value);
  const target   = document.getElementById('target-select').value;
  const algo1    = TASK_ALGOS[selectedTask].split(' + ')[0];
  const algo2    = TASK_ALGOS[selectedTask].split(' + ')[1];

  // Simulated progress log while the real API call runs
  const logs = [
    { t: '00:01', msg: 'Uploading dataset to backend…',             cls: 'info' },
    { t: '00:02', msg: 'Running preprocessing pipeline…',           cls: 'info' },
    { t: '00:04', msg: 'Handling missing values…',                  cls: 'info' },
    { t: '00:05', msg: 'Encoding categorical features…',            cls: 'info' },
    { t: '00:06', msg: 'Scaling numerical features…',               cls: 'info' },
    { t: '00:08', msg: `Splitting data ${splitPct}% / ${100-splitPct}%…`, cls: 'info' },
    { t: '00:10', msg: `Training model 1: ${algo1}…`,               cls: 'info' },
    { t: '00:15', msg: `Training model 2: ${algo2}…`,               cls: 'info' },
    { t: '00:18', msg: 'Evaluating both models on test set…',       cls: 'info' },
  ];

  let i = 0;
  const interval = setInterval(() => {
    if (i >= logs.length) { clearInterval(interval); return; }
    const l = logs[i];
    bar.style.width = Math.round((i + 1) / (logs.length + 2) * 85) + '%';
    const div = document.createElement('div');
    div.className = 'log-line';
    div.innerHTML = `<span class="ts">${l.t}</span><span class="${l.cls}">${l.msg}</span>`;
    logOut.appendChild(div);
    logOut.scrollTop = logOut.scrollHeight;
    i++;
  }, 350);

  // ── Real API call to /api/train ──
  fetch(`${API}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task:   selectedTask,
      target: target,
      split:  splitPct / 100,
    }),
  })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'Training failed'); });
      return r.json();
    })
    .then(data => {
      clearInterval(interval);
      bar.style.width = '100%';
      addLog(logOut, '✓', 'Best model selected — pipeline complete', 'ok');
      setTimeout(() => showResults(data), 400);
    })
    .catch(err => {
      clearInterval(interval);
      addLog(logOut, '✗', 'Error: ' + err.message, 'err');
      bar.style.width = '100%';
      bar.style.background = 'var(--red-text)';
      document.getElementById('run-btn').disabled = false;
    });
}

function addLog(container, ts, msg, cls) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<span class="ts">${ts}</span><span class="${cls}">${msg}</span>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

/* ─── Results ─── */
function showResults(data) {
  setStep(4);
  show('card-results');
  document.getElementById('card-results').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const best  = data.best_model;
  const other = data.other_model;

  // Primary metric label
  const metricLabel = { accuracy: 'Accuracy', r2: 'R²', silhouette_score: 'Silhouette' }[best.primary_metric] || best.primary_metric;

  document.getElementById('algo-compare-section').innerHTML = `
    <div class="algo-card winner">
      <div class="winner-badge">Selected</div>
      <div class="algo-name">${best.name}</div>
      <div class="algo-score">${metricLabel}: ${(best.score * 100).toFixed(1)}%</div>
    </div>
    <div class="algo-card">
      <div class="algo-name">${other.name}</div>
      <div class="algo-score">${metricLabel}: ${(other.score * 100).toFixed(1)}%</div>
    </div>`;

  document.getElementById('task-badge').textContent = selectedTask;
  buildMetrics(data.metrics);
  buildVisual(data.metrics);
  buildFeatureImportance(data.feature_importance);
  trainingDone = true;
}

function buildMetrics(metrics) {
  let cards = [];
  if (selectedTask === 'classification') {
    cards = [
      { label: 'Accuracy',  value: pct(metrics.accuracy)  },
      { label: 'Precision', value: pct(metrics.precision) },
      { label: 'Recall',    value: pct(metrics.recall)    },
      { label: 'F1-Score',  value: pct(metrics.f1)        },
    ];
  } else if (selectedTask === 'regression') {
    cards = [
      { label: 'R² Score', value: fmt(metrics.r2)  },
      { label: 'MAE',      value: fmt(metrics.mae) },
      { label: 'MSE',      value: fmt(metrics.mse) },
    ];
  } else {
    cards = [
      { label: 'Silhouette Score', value: fmt(metrics.silhouette_score) },
    ];
    if (metrics.n_clusters !== undefined) cards.push({ label: 'Clusters', value: metrics.n_clusters });
    if (metrics.inertia    !== undefined) cards.push({ label: 'Inertia',  value: fmt(metrics.inertia) });
  }
  document.getElementById('metrics-grid').innerHTML =
    cards.map(c => `<div class="metric-card"><div class="metric-label">${c.label}</div><div class="metric-value">${c.value}</div></div>`).join('');
}

function pct(v) { return v !== undefined && v !== null ? (parseFloat(v) * 100).toFixed(1) + '%' : '—'; }
function fmt(v) { return v !== undefined && v !== null ? parseFloat(v).toLocaleString(undefined, { maximumFractionDigits: 4 }) : '—'; }

function buildVisual(metrics) {
  const title   = document.getElementById('visual-title');
  const content = document.getElementById('visual-content');

  if (selectedTask === 'classification' && metrics.confusion_matrix) {
    title.textContent = 'Confusion matrix';
    const cm = metrics.confusion_matrix;
    if (cm.length === 2) {
      // Binary classification — 2×2 grid
      content.innerHTML = `
        <p style="font-size:11px;color:var(--text-tertiary);margin-bottom:6px">Predicted →</p>
        <div class="confusion-grid">
          <div class="cm-cell cm-label" style="font-size:10px">Actual →</div>
          <div class="cm-cell cm-label">Positive</div>
          <div class="cm-cell cm-label">Negative</div>
          <div class="cm-cell cm-label">Positive</div>
          <div class="cm-cell cm-tp">${cm[0][0]}<div class="cm-sub">TP</div></div>
          <div class="cm-cell cm-fn">${cm[0][1]}<div class="cm-sub">FN</div></div>
          <div class="cm-cell cm-label">Negative</div>
          <div class="cm-cell cm-fp">${cm[1][0]}<div class="cm-sub">FP</div></div>
          <div class="cm-cell cm-tn">${cm[1][1]}<div class="cm-sub">TN</div></div>
        </div>`;
    } else {
      // Multi-class: show as a simple table
      content.innerHTML = '<p style="font-size:13px;color:var(--text-secondary)">Multi-class confusion matrix:</p>' +
        '<div style="overflow-x:auto;margin-top:8px"><table style="font-size:11px">' +
        cm.map(row => '<tr>' + row.map(v => `<td style="padding:4px 8px;border:1px solid var(--border);text-align:center">${v}</td>`).join('') + '</tr>').join('') +
        '</table></div>';
    }
  } else if (selectedTask === 'regression') {
    title.textContent = 'Regression metrics summary';
    content.innerHTML = `<p style="font-size:13px;color:var(--text-secondary);margin-top:8px">
      Connect a chart library (Chart.js/Plotly) to plot actual vs. predicted or residuals.</p>
      <div class="api-note">GET /api/results/plot → returns chart data or base64 image</div>`;
  } else {
    title.textContent = 'Cluster visualization';
    content.innerHTML = `<p style="font-size:13px;color:var(--text-secondary);margin-top:8px">
      2D scatter (PCA-reduced) renders here.</p>
      <div class="api-note">GET /api/results/clusters → returns cluster assignments + PCA coords</div>`;
  }
}

function buildFeatureImportance(fi) {
  if (!fi || fi.length === 0) {
    document.getElementById('fi-section').innerHTML =
      '<p style="font-size:13px;color:var(--text-secondary)">Feature importance not available for this model type.</p>';
    return;
  }
  const maxImp = fi[0].importance;
  document.getElementById('fi-section').innerHTML =
    fi.map(f => `
      <div class="fi-row">
        <div class="fi-feature">${f.feature}</div>
        <div class="fi-bar-bg">
          <div class="fi-bar" style="width:${Math.round(f.importance / maxImp * 100)}%"></div>
        </div>
        <div class="fi-pct">${Math.round(f.importance * 100)}%</div>
      </div>`).join('');
}

/* ─── Save model ─── */
function saveModel() {
  if (!trainingDone) { alert('No trained model yet. Please run training first.'); return; }

  const btn = document.getElementById('save-btn');
  btn.textContent = 'Preparing download…';
  btn.disabled = true;

  // ── Real API call to /api/model/download ──
  fetch(`${API}/api/model/download`, { method: 'POST' })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'Download failed'); });
      return r.blob();
    })
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const a   = document.createElement('a');
      a.href     = url;
      a.download = 'best_model.pkl';
      a.click();
      URL.revokeObjectURL(url);
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" stroke-width="1.5" style="width:14px;height:14px;stroke:currentColor;fill:none">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5
             M12 3v13.5m0 0l-4.5-4.5M12 16.5l4.5-4.5"/></svg>
        Save model (.pkl)`;
    })
    .catch(err => {
      alert('Download error: ' + err.message);
      btn.disabled = false;
      btn.textContent = 'Save model (.pkl)';
    });
}
