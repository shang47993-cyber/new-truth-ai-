/* ============================================================
   Mysha — Frontend Application Logic (Waveform V2)
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    initAnalyze();
    initFeatureToggle();
    initSamples();
});

const SAMPLES = {
    ai: `The rapid advancement of artificial intelligence has fundamentally transformed how we approach problem-solving across industries. Machine learning algorithms now process vast datasets with unprecedented efficiency, enabling organizations to extract meaningful insights from complex information streams. This technological evolution represents a paradigm shift in computational capabilities, offering sophisticated tools for pattern recognition, natural language processing, and predictive analytics. The implications extend beyond mere automation, touching upon fundamental questions about the nature of intelligence itself and the future relationship between human cognition and artificial systems. As these technologies continue to mature, the boundary between human and machine capabilities becomes increasingly nuanced.`,
    
    human: `1983. “Don’t worry, I’ve got you!” Bright light cascades across my brother’s face like a benediction, his laughter scattering across the shore. I pray for his downfall in this silly little game more than I pray for his life. With an abandoned stick, we fashioned empires, an army of soldiers and sailors valiantly battling against whatever may come. “Were you even listening? Mon dieu, you are useless.” Dirt-bright eyes, barely level with my chin, stared me down as I knelt to greet them. “I’m truly very sorry,” I mock in a false apology. “To come find the whales with me.” I hesitate, if only for a moment. “I sibling swear. But you must lead the way, my brave pirate.” He beams as though I have given him something immeasurable. He is the only thing that feels like hope.`,
    
    story: `At exactly 11:42 every night, the last train arrived at Platform 3. People rushed onto it with practiced urgency: office workers loosening ties, students half-asleep over textbooks, tourists clutching maps they no longer needed. The station emptied within minutes, leaving behind only the echoes of footsteps and the faint smell of rain carried in through the open doors. Every night, Noah watched from the same bench. He was not waiting for the train. He was waiting for the girl with the yellow umbrella. She never boarded. Instead, she appeared just after the doors closed, always a few seconds too late. She would stop at the edge of the platform, look at the departing train, smile to herself, and disappear back up the stairs without a trace of frustration.`
};

function initSamples() {
    document.getElementById('sample-ai').addEventListener('click', () => {
        document.getElementById('text-input').value = SAMPLES.ai;
    });
    document.getElementById('sample-human').addEventListener('click', () => {
        document.getElementById('text-input').value = SAMPLES.human;
    });
    document.getElementById('sample-story').addEventListener('click', () => {
        document.getElementById('text-input').value = SAMPLES.story;
    });
}

function initAnalyze() {
    const btn = document.getElementById('btn-analyze');
    btn.addEventListener('click', runAnalysis);
}

async function runAnalysis() {
    const btn = document.getElementById('btn-analyze');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    const text = document.getElementById('text-input').value.trim();
    if (!text) {
        alert('Please enter or paste text to analyze.');
        return;
    }

    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');

    try {
        const resp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await resp.json();
        if (!resp.ok || data.error) {
            alert(data.error || 'Analysis execution failed.');
            return;
        }

        renderResults(data.analysis);
    } catch (err) {
        console.error(err);
        alert('Server connection error. Please ensure the backend is running.');
    } finally {
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
    }
}

function renderResults(analysis) {
    const section = document.getElementById('results-section');
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    renderDetection(analysis.detection);
    renderAuthorship(analysis.authorship);
    renderNeuralStats(analysis.detection.neural_metrics);
    renderWaveform(analysis.detection.neural_metrics);
    renderEvidence(analysis.detection.evidence);
    renderDendrogram(analysis.authorship);
    renderSilhouette(analysis.authorship);
    renderStyleShifts(analysis.authorship.style_shifts);
    renderFeatures(analysis.features);
}

function renderDetection(detection) {
    const gaugeDiv = document.getElementById('detection-gauge');
    const verdictDiv = document.getElementById('detection-verdict');
    const confidenceDiv = document.getElementById('detection-confidence');

    const humanPct = detection.human_probability;
    const aiPct = detection.ai_probability;

    Plotly.newPlot(gaugeDiv, [{
        type: 'indicator',
        mode: 'gauge+number',
        value: humanPct,
        number: {
            suffix: '% Human',
            font: { size: 18, color: '#e8e8f0' }
        },
        gauge: {
            axis: {
                range: [0, 100],
                tickfont: { color: '#6868a0', size: 10 },
                dtick: 25,
            },
            bar: { color: humanPct > 50 ? '#10b981' : '#ef4444', thickness: 0.6 },
            bgcolor: '#1a1a2e',
            borderwidth: 0,
            steps: [
                { range: [0, 35], color: 'rgba(239,68,68,0.2)' },
                { range: [35, 65], color: 'rgba(245,158,11,0.15)' },
                { range: [65, 100], color: 'rgba(16,185,129,0.2)' },
            ],
        }
    }], {
        margin: { t: 10, b: 5, l: 30, r: 30 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        height: 120,
        font: { color: '#e8e8f0' },
    }, { displayModeBar: false, responsive: true });

    verdictDiv.textContent = detection.verdict;
    confidenceDiv.innerHTML = `Confidence: <span class="confidence-badge ${detection.confidence}">${detection.confidence.toUpperCase()}</span>`;
}

function renderAuthorship(authorship) {
    const countDiv = document.getElementById('author-count');
    const verdictDiv = document.getElementById('authorship-verdict');
    const confidenceDiv = document.getElementById('authorship-confidence');

    const n = authorship.predicted_authors;
    countDiv.innerHTML = `
        <div class="number">${n}</div>
        <div class="label">${n === 1 ? 'Author Profile' : 'Distinct Author Styles'}</div>
    `;

    verdictDiv.textContent = authorship.verdict;
    confidenceDiv.innerHTML = `Clustering Confidence: <span class="confidence-badge ${authorship.confidence}">${authorship.confidence.toUpperCase()}</span>`;
}

function renderNeuralStats(metrics) {
    const grid = document.getElementById('neural-stats-grid');
    if (!metrics) {
        grid.innerHTML = '<p style="color: var(--text-muted);">Neural engine offline.</p>';
        return;
    }

    const items = [
        { label: 'Perplexity (PPL)', value: metrics.perplexity || '0' },
        { label: 'Mean Token LogProb', value: metrics.mean_log_prob || '0' },
        { label: 'High Certainty Density', value: `${((metrics.frac_low_entropy || 0) * 100).toFixed(1)}%` },
        { label: 'Waveform Volatility', value: metrics.wave_volatility || '0' },
    ];

    grid.innerHTML = items.map(item => `
        <div class="stat-item">
            <span class="stat-value">${item.value}</span>
            <span class="stat-label">${item.label}</span>
        </div>
    `).join('');
}

function renderWaveform(metrics) {
    const chartDiv = document.getElementById('waveform-chart');
    if (!metrics || !metrics.token_waveform || metrics.token_waveform.length === 0) {
        chartDiv.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No token waveform available.</p>';
        return;
    }

    const tokens = metrics.token_waveform.map(t => t.token);
    const logProbs = metrics.token_waveform.map(t => t.log_prob);

    Plotly.newPlot(chartDiv, [{
        x: tokens.map((_, i) => i),
        y: logProbs,
        text: tokens,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#6366f1', width: 2, shape: 'spline' },
        marker: { color: '#a78bfa', size: 5 },
        hovertemplate: '<b>%{text}</b><br>LogProb: %{y:.2f}<extra></extra>',
    }], {
        margin: { t: 15, b: 35, l: 45, r: 20 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: { text: 'Token Sequence Position', font: { color: '#6868a0', size: 10 } },
            color: '#6868a0',
            gridcolor: 'rgba(42,42,64,0.3)',
        },
        yaxis: {
            title: { text: 'Log Probability', font: { color: '#6868a0', size: 10 } },
            color: '#6868a0',
            gridcolor: 'rgba(42,42,64,0.3)',
        },
        font: { color: '#e8e8f0' },
        height: 220,
    }, { displayModeBar: false, responsive: true });
}

function renderEvidence(evidence) {
    const list = document.getElementById('evidence-list');
    if (!evidence || !evidence.length) {
        list.innerHTML = '<p style="color: var(--text-muted); padding: 1rem;">No decisive outlier signals detected.</p>';
        return;
    }

    list.innerHTML = evidence.map(e => `
        <div class="evidence-item ${e.direction.toLowerCase()}">
            <span class="evidence-direction ${e.direction.toLowerCase()}">${e.direction}</span>
            <div class="evidence-detail">
                <div class="evidence-signal">${e.description}</div>
                <div class="evidence-desc">${e.signal} | Impact: ${e.contribution}</div>
            </div>
            <span class="evidence-strength">${e.strength}</span>
        </div>
    `).join('');
}

function renderDendrogram(authorship) {
    const chartDiv = document.getElementById('dendrogram-chart');
    const segments = authorship.segments || [];

    if (segments.length < 2) {
        chartDiv.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">Requires 2+ text segments for tree linkage.</p>';
        return;
    }

    const clusterColors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
    const x = segments.map((_, i) => `Seg ${i + 1}`);
    const y = segments.map(s => s.features?.type_token_ratio || 0.5);
    const colors = segments.map(s => clusterColors[(s.cluster || 0) % clusterColors.length]);

    Plotly.newPlot(chartDiv, [{
        x, y,
        type: 'bar',
        marker: { color: colors, opacity: 0.85 },
        hovertemplate: '%{x}<br>TTR: %{y:.3f}<extra></extra>',
    }], {
        margin: { t: 10, b: 35, l: 45, r: 15 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { color: '#6868a0', gridcolor: 'rgba(42,42,64,0.3)' },
        yaxis: { title: { text: 'Lexical TTR', font: { color: '#6868a0', size: 10 } }, color: '#6868a0', gridcolor: 'rgba(42,42,64,0.3)' },
        font: { color: '#e8e8f0' },
        height: 240,
    }, { displayModeBar: false, responsive: true });
}

function renderSilhouette(authorship) {
    const chartDiv = document.getElementById('silhouette-chart');
    const scores = authorship.silhouette_scores || {};

    const ks = Object.keys(scores).map(Number);
    if (!ks.length) {
        chartDiv.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">Single author profile — no cluster variance.</p>';
        return;
    }

    const vals = ks.map(k => scores[k]);
    const bestK = authorship.predicted_authors;

    Plotly.newPlot(chartDiv, [{
        x: ks.map(k => `${k} Authors`),
        y: vals,
        type: 'bar',
        marker: { color: ks.map(k => k === bestK ? '#6366f1' : '#2a2a40') },
        hovertemplate: '%{x}: Silhouette %{y:.3f}<extra></extra>',
    }], {
        margin: { t: 10, b: 35, l: 45, r: 15 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { color: '#6868a0', gridcolor: 'rgba(42,42,64,0.3)' },
        yaxis: { title: { text: 'Silhouette Score', font: { color: '#6868a0', size: 10 } }, color: '#6868a0', gridcolor: 'rgba(42,42,64,0.3)' },
        font: { color: '#e8e8f0' },
        height: 240,
    }, { displayModeBar: false, responsive: true });
}

function renderStyleShifts(shifts) {
    const section = document.getElementById('style-shifts-section');
    const list = document.getElementById('style-shifts-list');

    if (!shifts || shifts.length === 0) {
        section.classList.add('hidden');
        return;
    }
    section.classList.remove('hidden');

    list.innerHTML = shifts.map(s => `
        <div class="shift-item">
            <span class="shift-badge">Δ ${s.shift_magnitude}σ</span>
            <span>${s.description}</span>
        </div>
    `).join('');
}

function renderFeatures(features) {
    const table = document.getElementById('features-table');
    if (!features) return;

    const rows = Object.entries(features)
        .filter(([k]) => k !== 'error')
        .map(([key, val]) => `
            <tr>
                <td>${key.replace(/_/g, ' ')}</td>
                <td>${typeof val === 'number' ? val.toLocaleString(undefined, { maximumFractionDigits: 4 }) : val}</td>
            </tr>
        `).join('');

    table.innerHTML = `
        <table>
            <thead><tr><th>Linguistic Dimension</th><th>Mathematical Value</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function initFeatureToggle() {
    const btn = document.getElementById('toggle-features');
    const table = document.getElementById('features-table');
    btn.addEventListener('click', () => {
        table.classList.toggle('hidden');
        btn.textContent = table.classList.contains('hidden')
            ? 'Toggle Full Feature Table ▼'
            : 'Hide Feature Table ▲';
    });
}
