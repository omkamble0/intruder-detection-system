// --- TAB STATE MANAGEMENT ---
document.addEventListener("DOMContentLoaded", () => {
    let savedTab = sessionStorage.getItem('activeTab');
    if (savedTab) { switchTab(savedTab); }
});

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('tab-active'));
    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('bg-cyan-500/20', 'neon-text', 'neon-border');
        btn.classList.add('hover:bg-slate-800', 'text-slate-400', 'border-transparent');
    });
    
    document.getElementById('tab-' + tabId).classList.add('tab-active');
    let activeBtn = document.getElementById('btn-' + tabId);
    if(activeBtn) {
        activeBtn.classList.remove('hover:bg-slate-800', 'text-slate-400', 'border-transparent');
        activeBtn.classList.add('bg-cyan-500/20', 'neon-text', 'border', 'neon-border');
    }
    sessionStorage.setItem('activeTab', tabId);
}

// --- IMAGE MODAL LOGIC ---
function openModal(imageSrc) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    modalImg.src = imageSrc;
    modal.classList.remove('hidden');
    setTimeout(() => modal.classList.remove('opacity-0'), 10);
}

function closeModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.add('opacity-0');
    setTimeout(() => modal.classList.add('hidden'), 300);
}

// --- CORE API FUNCTIONS ---
function updateRateLimit(seconds) {
    fetch('/api/config', { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ cooldown: seconds }) 
    });
}

function toggleCamera() {
    fetch('/api/camera', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        const btn = document.getElementById('camera-btn');
        if(data.status === 'active') {
            btn.innerHTML = '⏹ Stop Node';
            btn.className = 'px-6 py-2 rounded-lg font-bold text-white transition bg-red-600/80 hover:bg-red-500 shadow-[0_0_15px_rgba(220,38,38,0.5)] border border-red-400';
        } else {
            btn.innerHTML = '▶ Start Node';
            btn.className = 'px-6 py-2 rounded-lg font-bold text-white transition bg-cyan-600/80 hover:bg-cyan-500 shadow-[0_0_15px_rgba(0,242,254,0.5)] border border-cyan-400';
        }
    });
}

function deleteLog(logId) {
    if(!confirm("Permenantly Delete this threat evidence from the secure database?")) return;
    
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
