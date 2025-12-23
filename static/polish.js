/* VCAADecode Admin Polish JS */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Toast Container
    const toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    document.body.appendChild(toastContainer);

    // --- Toast System ---
    window.showToast = function(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'ℹ️';
        if (type === 'success') icon = '✅';
        if (type === 'error') icon = '⚠️';
        
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <span class="toast-message">${message}</span>
        `;
        
        toastContainer.appendChild(toast);
        
        // Trigger animation
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });
        
        // Auto dismiss
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    };

    // --- Quick Actions Handlers ---
    
    // 1. Refresh Data
    const refreshBtn = document.getElementById('action-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            
            // Show loading state on metrics
            const metrics = document.querySelectorAll('.metric-value');
            const originalValues = [];
            metrics.forEach((el, i) => {
                originalValues[i] = el.innerText;
                el.classList.add('skeleton', 'skeleton-text');
                el.innerText = '';
            });

            try {
                const response = await fetch('/api/stats/live?key=' + getAdminKey());
                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                
                // Update DOM
                if (data.headline) {
                    document.getElementById('metric-sessions').innerText = data.headline.total_sessions;
                    document.getElementById('metric-questions').innerText = data.headline.questions_24h;
                    document.getElementById('metric-users').innerText = data.headline.active_users_7d;
                }
                
                showToast('Dashboard data refreshed', 'success');
            } catch (err) {
                console.error('Refresh failed:', err);
                showToast('Failed to refresh data', 'error');
                
                // Restore values
                metrics.forEach((el, i) => {
                    el.innerText = originalValues[i];
                });
            } finally {
                // Remove skeleton classes
                metrics.forEach(el => {
                    el.classList.remove('skeleton', 'skeleton-text');
                });
            }
        });
    }

    // 2. Export CSV
    const exportBtn = document.getElementById('action-export');
    if (exportBtn) {
        exportBtn.addEventListener('click', function(e) {
            // Let default link behavior happen, but show toast
            showToast('Exporting data to CSV...', 'info');
        });
    }

    // 3. Help Modal
    const helpBtn = document.getElementById('action-help');
    if (helpBtn) {
        helpBtn.addEventListener('click', function() {
            showToast('Admin documentation coming soon!', 'info');
        });
    }

    // Helper to get key from URL
    function getAdminKey() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('key') || '';
    }

    // Check for URL-based toast triggers (e.g. after redirect)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('toast')) {
        const msg = urlParams.get('toast');
        const type = urlParams.get('type') || 'info';
        showToast(decodeURIComponent(msg), type);
        
        // Clean URL
        const newUrl = window.location.pathname + '?key=' + getAdminKey();
        window.history.replaceState({}, document.title, newUrl);
    }
});
