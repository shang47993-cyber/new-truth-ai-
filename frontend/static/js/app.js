/* ============================================================
   Mysha — Frontend Application Logic (Waveform V2.5 + Upload)
   ============================================================ */

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initFileUpload();
    initAnalyze();
    initFeatureToggle();
    initSamples();
});

const SAMPLES = {
    ai: `The rapid advancement of artificial intelligence has fundamentally transformed how we approach problem-solving across industries. Machine learning algorithms now process vast datasets with unprecedented efficiency, enabling organizations to extract meaningful insights from complex information streams. This technological evolution represents a paradigm shift in computational capabilities, offering sophisticated tools for pattern recognition, natural language processing, and predictive analytics. The implications extend beyond mere automation, touching upon fundamental questions about the nature of intelligence itself and the future relationship between human cognition and artificial systems. As these technologies continue to mature, the boundary between human and machine capabilities becomes increasingly nuanced.`,
    
    human: `1983. “Don’t worry, I’ve got you!” Bright light cascades across my brother’s face like a benediction, his laughter scattering across the shore. I pray for his downfall in this silly little game more than I pray for his life. With an abandoned stick, we fashioned empires, an army of soldiers and sailors valiantly battling against whatever may come. “Were you even listening? Mon dieu, you are useless.” Dirt-bright eyes, barely level with my chin, stared me down as I knelt to greet them. “I’m truly very sorry,” I mock in a false apology. “To come find the whales with me.” I hesitate, if only for a moment. “I sibling swear. But you must lead the way, my brave pirate.” He beams as though I have given him something immeasurable. He is the only thing that feels like hope.`,
    
    story: `At exactly 11:42 every night, the last train arrived at Platform 3. People rushed onto it with practiced urgency: office workers loosening ties, students half-asleep over textbooks, tourists clutching maps they no longer needed. The station emptied within minutes, leaving behind only the echoes of footsteps and the faint smell of rain carried in through the open doors. Every night, Noah watched from the same bench. He was not waiting for the train. He was waiting for the girl with the yellow umbrella. She never boarded. Instead, she appeared just after the doors closed, always a few seconds too late. She would stop at the edge of the platform, look at the departing train, smile to herself, and disappear back up the stairs without a trace of frustration.`
};

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });
}

function initFileUpload() {
    const dropzone = document.getElementById('file-dropzone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const previewBar = document.getElementById('file-preview-bar');
    const fileNameSpan = document.getElementById('selected-file-name');
    const fileSizeSpan = document.getElementById('selected-file-size');
    const btnRemove = document.getElementById('btn-remove-file');

    btnBrowse.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
        if (e.target !== btnBrowse) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    btnRemove.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewBar.classList.add('hidden');
        dropzone.classList.remove('hidden');
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameSpan.textContent = file.name;
        fileSizeSpan.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
        previewBar.classList.remove('hidden');
        dropzone.classList.add('hidden');
    }
}

function initSamples() {
    document.getElementById('sample-ai').addEventListener('click', () => {
        document.getElementById('tab-btn-text').click();
        document.getElementById('text-input').value = SAMPLES.ai;
    });
    document.getElementById('sample-human').addEventListener('click', () => {
        document.getElementById('tab-btn-text').click();
        document.getElementById('text-input').value = SAMPLES.human;
    });
    document.getElementById('sample-story').addEventListener('click', () => {
        document.getElementById('tab-btn-text').click();
        document.getElementById('text-input').value = SAMPLES.story;
    });
}

function initFeatureToggle() {
    const btn = document.getElementById('btn-toggle-features');
    const grid = document.getElementById('features-grid');
    if (!btn || !grid) return;
    btn.addEventListener('click', () => {
        const isHidden = grid.classList.toggle('hidden');
        btn.textContent = isHidden ? 'Expand Feature Vector Matrix ▾' : 'Collapse Feature Vector Matrix ▴';
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

    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');

    let endpoint = '/api/analyze';
    let requestOptions = {};

    if (activeTab === 'upload') {
        if (!selectedFile) {
            alert('Please select or drop a document (PDF, DOCX, TXT, CSV) to analyze.');
            return;
        }
        endpoint = '/api/upload';
        const formData = new FormData();
        formData.append('file', selectedFile);
        requestOptions = {
            method: 'POST',
            body: formData
        };
    } else {
        const text = document.getElementById('text-input').value.trim();
        if (!text) {
            alert('Please enter or paste text to analyze.');
            return;
        }
        requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        };
    }

    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');

    try {
        const resp = await fetch(endpoint, requestOptions);
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

    // Source Banner
    const sourceLabel = document.getElementById('source-label');
    if (sourceLabel) {
        sourceLabel.textContent = `Source: ${analysis.source || 'Pasted Text'} — ${analysis.word_count || 0} words analyzed`;
    }

    renderDetection(analysis.detection);
    renderAuthorship(analysis.authorship);
    renderNeuralStats(analysis.detection.neural_metrics);
    renderSentenceHeatmap(analysis.detection.sentence_analysis);
    renderWaveform(analysis.detection.neural_metrics);
    renderEvidence(analysis.detection.evidence);
    renderDendrogram(analysis.authorship);
    renderSilhouette(analysis.authorship);
    renderStyleShifts(analysis.authorship.style_shifts);
    renderFeatures(analysis.features);
}

function renderDetection(detection) {
    const verdictEl = document.getElementById('detection-verdict');
    const confEl = document.getElementById('detection-confidence');

    const aiProb = Math.round(detection.ai_probability);
    const humanProb = Math.round(detection.human_probability);
    const verdict = detection.verdict;

    let verdictColor = '#f59e0b';
    if (aiProb >= 65) verdictColor = '#ef4444';
    else if (humanProb >= 65) verdictColor = '#10b981';

    verdictEl.innerHTML = `<span style="color: ${verdictColor}">${verdict}</span>`;
    confEl.innerHTML = `Confidence: <strong>${detection.confidence.toUpperCase()}</strong> | AI: ${aiProb}% — Human: ${humanProb}%`;

    const gaugeData = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: aiProb,
        number: { suffix: '% AI', font: { color: '#e8e8f0', size: 24 } },
        gauge: {
            axis: { range: [0, 100], tickcolor: '#9898b0', tickfont: { color: '#9898b0' } },
            bar: { color: aiProb > 50 ? '#ef4444' : '#10b981', thickness: 0.3 },
            bgcolor: '#0f0f1a',
            borderwidth: 1,
            bordercolor: '#2a2a40',
            steps: [
                { range: [0, 40], color: 'rgba(16, 185, 129, 0.2)' },
                { range: [40, 60], color: 'rgba(245, 158, 11, 0.2)' },
                { range: [60, 100], color: 'rgba(239, 68, 68, 0.2)' }
            ]
        }
    }];

    const layout = {
        margin: { t: 20, b: 10, l: 30, r: 30 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        height: 150
    };

    Plotly.newPlot('detection-gauge', gaugeData, layout, { displayModeBar: false, responsive: true });
}

function renderAuthorship(authorship) {
    document.getElementById('author-count').textContent = authorship.predicted_authors;
    document.getElementById('authorship-verdict').textContent = authorship.verdict;
    document.getElementById('authorship-confidence').innerHTML = `Confidence: <strong>${(authorship.confidence || 'medium').toUpperCase()}</strong>`;
}

function renderNeuralStats(neural) {
    const grid = document.getElementById('neural-stats-grid');
    if (!neural) {
        grid.innerHTML = '<div class="stat-item"><span class="stat-label">Neural Engine</span><span class="stat-value">N/A</span></div>';
        return;
    }

    grid.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Model Perplexity</span>
            <span class="stat-value">${neural.perplexity || 0}</span>
            <span class="stat-sub">&lt; 30 = High Predictability</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">GLTR Top-10 Predictable</span>
            <span class="stat-value">${Math.round((neural.gltr_top10 || 0) * 100)}%</span>
            <span class="stat-sub">&gt; 50% = AI Pattern</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Wave Volatility</span>
            <span class="stat-value">${neural.wave_volatility || 0}</span>
            <span class="stat-sub">Human writing has high variance</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">GLTR Rare Words (&gt;1k)</span>
            <span class="stat-value">${Math.round((neural.gltr_tail1000 || 0) * 100)}%</span>
            <span class="stat-sub">Creative/Uncommon vocabulary</span>
        </div>
    `;
}

function renderSentenceHeatmap(sentences) {
    const card = document.getElementById('sentence-heatmap-card');
    const container = document.getElementById('sentence-heatmap');
    if (!sentences || sentences.length === 0) {
        card.classList.add('hidden');
        return;
    }
    card.classList.remove('hidden');

    container.innerHTML = sentences.map(s => {
        let bgStyle = '';
        let badgeColor = '';
        if (s.tag === 'ai') {
            bgStyle = 'background: rgba(239, 68, 68, 0.15); border-left: 3px solid #ef4444;';
            badgeColor = 'color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4);';
        } else if (s.tag === 'human') {
            bgStyle = 'background: rgba(16, 185, 129, 0.15); border-left: 3px solid #10b981;';
            badgeColor = 'color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);';
        } else {
            bgStyle = 'background: rgba(245, 158, 11, 0.15); border-left: 3px solid #f59e0b;';
            badgeColor = 'color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4);';
        }

        return `
            <div style="${bgStyle} padding: 0.75rem 1rem; margin-bottom: 0.6rem; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; gap: 1rem;">
                <span style="color: #e8e8f0; font-size: 0.92rem; line-height: 1.4;">${s.sentence}</span>
                <span style="${badgeColor} font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 12px; white-space: nowrap; font-weight: 600;">
                    ${s.ai_probability}% AI
                </span>
            </div>
        `;
    }).join('');
}

function renderWaveform(neural) {
    const chartDiv = document.getElementById('waveform-chart');
    if (!neural || !neural.token_details || neural.token_details.length === 0) {
        chartDiv.innerHTML = '<p style="color: #9898b0; padding: 1rem;">Waveform metrics not available for short text.</p>';
        return;
    }

    const tokens = neural.token_details.map(t => t.token);
    const logProbs = neural.token_details.map(t => t.log_prob);
    const colors = neural.token_details.map(t => {
        if (t.rank < 10) return '#10b981';   // Green (Top 10)
        if (t.rank < 100) return '#f59e0b';  // Yellow (Top 100)
        return '#ef4444';                    // Red (Tail)
    });

    const trace = {
        x: tokens.map((_, i) => i),
        y: logProbs,
        text: tokens,
        mode: 'lines+markers',
        marker: { size: 8, color: colors },
        line: { color: '#6366f1', width: 2, shape: 'spline' },
        type: 'scatter',
        hovertemplate: '<b>Token:</b> %{text}<br><b>Log Prob:</b> %{y:.2f}<extra></extra>'
    };

    const layout = {
        margin: { t: 10, b: 30, l: 40, r: 20 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { showgrid: false, zeroline: false, tickfont: { color: '#9898b0' } },
        yaxis: { title: 'Log Probability', titlefont: { color: '#9898b0', size: 11 }, showgrid: true, gridcolor: '#2a2a40', tickfont: { color: '#9898b0' } }
    };

    Plotly.newPlot('waveform-chart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderEvidence(evidence) {
    const list = document.getElementById('evidence-list');
    if (!evidence || evidence.length === 0) {
        list.innerHTML = '<p style="color: #9898b0;">No conclusive statistical anomalies detected.</p>';
        return;
    }

    list.innerHTML = evidence.map(e => `
        <div class="evidence-item">
            <span class="evidence-signal">${e.signal || 'Signal'}</span>
            <span class="evidence-detail">${e.detail || ''}</span>
        </div>
    `).join('');
}

function renderDendrogram(authorship) {
    const chartDiv = document.getElementById('dendrogram-chart');
    if (!authorship.dendrogram_data) {
        chartDiv.innerHTML = '<p style="color: #9898b0; padding: 1rem; text-align: center;">Single writing cluster detected across segments.</p>';
        return;
    }

    const data = authorship.dendrogram_data;
    const traces = [];
    for (let i = 0; i < data.icoord.length; i++) {
        traces.push({
            x: data.dcoord[i],
            y: data.icoord[i],
            mode: 'lines',
            line: { color: data.color_list ? data.color_list[i] : '#a78bfa', width: 2 },
            type: 'scatter',
            hoverinfo: 'none'
        });
    }

    const layout = {
        showlegend: false,
        margin: { t: 10, b: 20, l: 30, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { showgrid: true, gridcolor: '#2a2a40', tickfont: { color: '#9898b0' } },
        yaxis: { showgrid: false, showticklabels: false }
    };

    Plotly.newPlot('dendrogram-chart', traces, layout, { displayModeBar: false, responsive: true });
}

function renderSilhouette(authorship) {
    const chartDiv = document.getElementById('silhouette-chart');
    if (!authorship.candidate_evaluations || authorship.candidate_evaluations.length === 0) {
        chartDiv.innerHTML = '<p style="color: #9898b0; padding: 1rem; text-align: center;">Uniform authorship profile across text chunks.</p>';
        return;
    }

    const evals = authorship.candidate_evaluations;
    const trace = {
        x: evals.map(e => `k=${e.k}`),
        y: evals.map(e => e.silhouette_score),
        type: 'bar',
        marker: { color: evals.map(e => e.k === authorship.predicted_authors ? '#6366f1' : '#2a2a40') }
    };

    const layout = {
        margin: { t: 10, b: 30, l: 40, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { tickfont: { color: '#9898b0' } },
        yaxis: { range: [-0.2, 1.0], gridcolor: '#2a2a40', tickfont: { color: '#9898b0' } }
    };

    Plotly.newPlot('silhouette-chart', [trace], layout, { displayModeBar: false, responsive: true });
}

function renderStyleShifts(shifts) {
    const list = document.getElementById('style-shifts-list');
    if (!shifts || shifts.length === 0) {
        list.innerHTML = '<p style="color: #9898b0; font-size: 0.9rem;">No major stylistic breaks or register shifts detected.</p>';
        return;
    }

    list.innerHTML = shifts.map(s => `
        <div style="background: var(--bg-input); border-left: 3px solid #f59e0b; padding: 0.6rem 1rem; margin-bottom: 0.5rem; border-radius: 4px;">
            <div style="font-weight: 600; font-size: 0.85rem; color: #f59e0b;">Style Shift at Boundary: ${s.boundary || 'Section'}</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem;">Magnitude: ${s.distance || '0.0'}σ distance deviation</div>
        </div>
    `).join('');
}

function renderFeatures(features) {
    const grid = document.getElementById('features-grid');
    if (!features) return;

    grid.innerHTML = Object.entries(features).map(([k, v]) => `
        <div class="feature-box">
            <div class="feature-key">${k.replace(/_/g, ' ').toUpperCase()}</div>
            <div class="feature-val">${v}</div>
        </div>
    `).join('');
}
