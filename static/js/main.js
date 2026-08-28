// --- SIDEBAR TOGGLE (HAMBURGER MENU) ---
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar.classList.contains('-translate-x-full')) {
        sidebar.classList.remove('-translate-x-full');
    } else {
        sidebar.classList.add('-translate-x-full');
    }
}

// --- TAB STATE MANAGEMENT ---
document.addEventListener("DOMContentLoaded", () => {
    let savedTab = sessionStorage.getItem('activeTab');
    if (savedTab) { switchTab(savedTab); }
    else { switchTab('dashboard'); } // Default to dashboard
});

function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
        tab.classList.remove('tab-active');
    });
    
    // Reset all buttons
    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('bg-cyan-500/20', 'neon-text', 'neon-border');
        btn.classList.add('hover:bg-slate-800', 'text-slate-400', 'border-transparent');
    });
    
    // Show active tab
    const activeTab = document.getElementById('tab-' + tabId);
    if (activeTab) {
        activeTab.classList.remove('hidden');
        activeTab.classList.add('tab-active');
    }
    
    // Highlight active button
    let activeBtn = document.getElementById('btn-' + tabId);
    if(activeBtn) {
        activeBtn.classList.remove('hover:bg-slate-800', 'text-slate-400', 'border-transparent');
        activeBtn.classList.add('bg-cyan-500/20', 'neon-text', 'border', 'neon-border');
    }
    
    sessionStorage.setItem('activeTab', tabId);
    
    // Auto-close sidebar on mobile after clicking
    if (window.innerWidth < 1024) {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.add('-translate-x-full');
    }
}

// --- CAMERA SOURCE CONTROLS ---
function toggleCameraInput() {
    const type = document.getElementById('camera-type').value;
    const urlInput = document.getElementById('camera-url');
    if (type === 'remote') {
        urlInput.classList.remove('hidden');
    } else {
        urlInput.classList.add('hidden');
    }
}

function toggleCamera() {
    const btn = document.getElementById('camera-btn');
    const isStarting = btn.innerHTML.includes('Start Node');
    
    const type = document.getElementById('camera-type').value;
    const url = document.getElementById('camera-url').value;

    // Prevent starting remote camera if no URL is provided
    if (type === 'remote' && isStarting && !url) {
        alert("Please enter your Phone's IP Camera URL first.");
        return;
    }

    fetch('/api/camera', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            action: isStarting ? 'start' : 'stop',
            type: type,
            url: url
        })
    })
    .then(res => res.json())
    .then(data => {
        if(data.status === 'active') {
            btn.innerHTML = '⏹ Stop Node';
            btn.className = 'px-6 py-3 h-12 rounded-lg font-bold text-white transition bg-red-600/80 hover:bg-red-500 shadow-[0_0_15px_rgba(220,38,38,0.5)] border border-red-400 whitespace-nowrap';
        } else {
            btn.innerHTML = '▶ Start Node';
            btn.className = 'px-6 py-3 h-12 rounded-lg font-bold text-white transition bg-cyan-600/80 hover:bg-cyan-500 shadow-[0_0_15px_rgba(0,242,254,0.5)] border border-cyan-400 whitespace-nowrap';
        }
    });
}

// --- THREAT SANDBOX LOGIC (LOCAL ENGINE) ---
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const dropZone = document.getElementById('drop-zone');
    
    // Show Loading State
    dropZone.innerHTML = `<div class="animate-pulse text-cyan-400 text-center"><span class="text-5xl block mb-4 animate-spin">⚙️</span><p class="font-mono">Running Local Static Analysis...</p></div>`;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/scan_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Reset Drop Zone UI
        dropZone.innerHTML = `<input type="file" id="file-input" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" onchange="handleFileUpload(event)">
            <span class="text-5xl mb-4">📂</span>
            <p class="text-slate-300 font-semibold text-lg">Drop suspicious payloads here</p>
            <p class="text-slate-500 text-sm mt-2 uppercase tracking-widest">or click to browse local drive</p>`;

        // Populate the Report Card
        document.getElementById('scan-results').classList.remove('hidden');
        document.getElementById('scan-filename').innerText = "TARGET: " + (data.filename || file.name);
        document.getElementById('scan-hash').innerText = "SHA-256: " + (data.hash || "ERROR");
        
        const statsDiv = document.getElementById('scan-stats');
        
        if (data.status === 'success') {
            let threatColor = data.threat_level === 'MALICIOUS' ? 'red' : (data.threat_level === 'SUSPICIOUS' ? 'yellow' : 'green');
            
            statsDiv.innerHTML = `
                <div class="bg-${threatColor}-500/10 border border-${threatColor}-500/30 p-4 rounded text-center col-span-2 md:col-span-1 shadow-[0_0_15px_rgba(0,0,0,0.2)]">
                    <p class="text-xl font-bold text-${threatColor}-500">${data.threat_level}</p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-1">Classification</p>
                </div>
                <div class="bg-slate-800/50 border border-slate-700 p-4 rounded text-center">
                    <p class="text-2xl font-bold text-cyan-400">${data.stats.entropy} <span class="text-xs text-slate-500">/ 8.0</span></p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-1">Shannon Entropy</p>
                </div>
                <div class="bg-slate-800/50 border border-slate-700 p-4 rounded text-center">
                    <p class="text-2xl font-bold text-red-400">${data.stats.malicious}</p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-1">Risk Factors</p>
                </div>
            `;
        } else {
            statsDiv.innerHTML = `<div class="col-span-4 bg-red-500/10 border border-red-500/50 p-4 rounded"><p class="text-red-400 font-mono text-sm">${data.error}</p></div>`;
        }
    })
    .catch(err => {
        dropZone.innerHTML = `<p class="text-red-500 font-bold font-mono">ANALYSIS ENGINE FAILED.</p>`;
    });
}

// --- LEGACY UTILITY FUNCTIONS (For Config/Audit Tabs) ---
function openModal(imageSrc) {
    const modal = document.getElementById('image-modal');
    if (!modal) return;
    const modalImg = document.getElementById('modal-img');
    modalImg.src = imageSrc;
    modal.classList.remove('hidden');
    setTimeout(() => modal.classList.remove('opacity-0'), 10);
}

function closeModal() {
    const modal = document.getElementById('image-modal');
    if (!modal) return;
    modal.classList.add('opacity-0');
    setTimeout(() => modal.classList.add('hidden'), 300);
}

function updateRateLimit(seconds) {
    fetch('/api/config', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ cooldown: seconds }) 
    });
}

function deleteLog(logId) {
    if(!confirm("Permanently Delete this threat evidence from the secure database?")) return;
    
    fetch('/api/logs/' + logId, { method: 'DELETE' })
    .then(res => res.json())
    .then(data => {
        if(data.status === 'success') {
            let rowToDelete = document.getElementById('log-row-' + logId);
            if (rowToDelete) {
                rowToDelete.style.transition = "opacity 0.3s ease, transform 0.3s ease";
                rowToDelete.style.opacity = "0";
                rowToDelete.style.transform = "translateX(-20px)";
                setTimeout(() => rowToDelete.remove(), 300);
            }
        } else {
            alert("System Error: Unable to Delete log.");
        }
    });
}

// --- WEB THREAT INTELLIGENCE LOGIC ---
function scanUrl() {
    const urlInput = document.getElementById('url-input').value.trim();
    if (!urlInput) {
        alert("Please enter a URL to scan.");
        return;
    }

    // Show the results panel and loading animation
    const resultsPanel = document.getElementById('url-results');
    const loadingDiv = document.getElementById('url-loading');
    const statsDiv = document.getElementById('url-stats');
    
    resultsPanel.classList.remove('hidden');
    loadingDiv.classList.remove('hidden');
    statsDiv.classList.add('hidden');

    // Send the URL to the Python Backend
    fetch('/api/scan_url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: urlInput })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading, show stats
        loadingDiv.classList.add('hidden');
        statsDiv.classList.remove('hidden');
        
        if (data.status === 'success') {
            let threatColor = data.result === 'PHISHING / MALICIOUS' ? 'red' : 'green';
            let icon = data.result === 'PHISHING / MALICIOUS' ? '⚠️' : '✅';
            
            statsDiv.innerHTML = `
                <div class="md:col-span-3 bg-black/40 p-4 rounded border border-slate-700 mb-2">
                    <p class="text-xs text-slate-500 uppercase tracking-widest mb-1">Target Payload</p>
                    <p class="font-mono text-sm text-white break-all">${data.url}</p>
                </div>

                <div class="bg-${threatColor}-500/10 border border-${threatColor}-500/30 p-6 rounded-xl text-center shadow-[0_0_20px_rgba(0,0,0,0.3)]">
                    <span class="text-4xl block mb-2">${icon}</span>
                    <p class="text-2xl font-bold text-${threatColor}-500">${data.result}</p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-1">AI Classification</p>
                </div>
                
                <div class="bg-slate-800/50 border border-slate-700 p-6 rounded-xl text-center">
                    <p class="text-3xl font-bold text-cyan-400 mt-2">${(data.confidence * 100).toFixed(1)}%</p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-2">Model Confidence</p>
                </div>
                
                <div class="bg-slate-800/50 border border-slate-700 p-6 rounded-xl text-center">
                    <p class="text-3xl font-bold text-slate-300 mt-2">NLP</p>
                    <p class="text-xs text-slate-400 uppercase tracking-widest mt-2">Detection Engine</p>
                </div>
            `;
        } else {
            statsDiv.innerHTML = `<div class="md:col-span-3 bg-red-500/10 border border-red-500/50 p-4 rounded"><p class="text-red-400 font-mono text-sm">Error: ${data.message}</p></div>`;
        }
    })
    .catch(err => {
        loadingDiv.classList.add('hidden');
        statsDiv.classList.remove('hidden');
        statsDiv.innerHTML = `<div class="md:col-span-3 bg-red-500/10 border border-red-500/50 p-4 rounded"><p class="text-red-400 font-mono font-bold text-sm">CONNECTION TO AI BRAIN FAILED.</p></div>`;
    });
}