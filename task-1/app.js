// State management
let allModels = [];
let filteredModels = [];
let selectedModel = null;

// DOM elements
const modelSelect = document.getElementById('modelSelect');
const modelSearch = document.getElementById('modelSearch');
const modelCount = document.getElementById('modelCount');
const specsContainer = document.getElementById('specsContainer');
const actionSection = document.getElementById('actionSection');
const resultsSection = document.getElementById('resultsSection');
const findHardwareBtn = document.getElementById('findHardwareBtn');
const resultsContent = document.getElementById('resultsContent');

// Initialize the application
async function init() {
    await loadModels();
    setupEventListeners();
}

// Load models from HuggingFace API
async function loadModels() {
    try {
        // Show loading state
        modelCount.innerHTML = '<div class="loading-dot"></div> Loading models...';

        const response = await fetch('/api/models');
        const data = await response.json();

        allModels = data.models || [];
        filteredModels = [...allModels];

        populateModelSelect(filteredModels);
        updateModelCount(filteredModels.length);

    } catch (error) {
        console.error('Error loading models:', error);
        modelCount.textContent = 'Error loading models';
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Populate the model select dropdown
function populateModelSelect(models) {
    if (models.length === 0) {
        modelSelect.innerHTML = '<option value="">No models found</option>';
        return;
    }

    const options = models.map(model => {
        const displayName = model.modelId || model.id;
        return `<option value="${model.id}">${displayName}</option>`;
    }).join('');

    modelSelect.innerHTML = '<option value="">Select a model...</option>' + options;
}

// Update model count display
function updateModelCount(count) {
    modelCount.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5" opacity="0.5"/>
            <circle cx="8" cy="8" r="2" fill="currentColor"/>
        </svg>
        ${count.toLocaleString()} models available
    `;
}

// Setup event listeners
function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Cmd/Ctrl + K to focus search
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            modelSearch.focus();
        }

        // Cmd/Ctrl + Enter to trigger hardware search
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && selectedModel) {
            e.preventDefault();
            findHardwareBtn.click();
        }

        // Escape to clear search
        if (e.key === 'Escape' && document.activeElement === modelSearch) {
            modelSearch.value = '';
            modelSearch.dispatchEvent(new Event('input'));
        }
    });

    // Model search
    modelSearch.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase().trim();

        if (searchTerm === '') {
            filteredModels = [...allModels];
        } else {
            filteredModels = allModels.filter(model => {
                const modelId = (model.modelId || model.id || '').toLowerCase();
                const author = (model.author || '').toLowerCase();
                return modelId.includes(searchTerm) || author.includes(searchTerm);
            });
        }

        populateModelSelect(filteredModels);
        updateModelCount(filteredModels.length);
    });

    // Model selection
    modelSelect.addEventListener('change', async (e) => {
        const modelId = e.target.value;

        if (!modelId) {
            showEmptyState();
            return;
        }

        const model = allModels.find(m => m.id === modelId);
        if (model) {
            selectedModel = model;
            await displayModelSpecs(model);
            showActionSection();
        }
    });

    // Find hardware button
    findHardwareBtn.addEventListener('click', async () => {
        if (!selectedModel) return;

        await findHardware(selectedModel);
    });
}

// Display model specifications
async function displayModelSpecs(model) {
    // Fetch detailed model info
    try {
        const response = await fetch(`/api/model-details/${encodeURIComponent(model.id)}`);
        const details = await response.json();

        const specs = {
            parameters: details.parameters || 'Unknown',
            ram: details.ram || 'Calculating...',
            vram: details.vram || 'Calculating...',
            type: details.type || 'Unknown',
            precision: details.precision || 'FP16'
        };

        specsContainer.innerHTML = `
            <div class="specs-grid">
                <div class="spec-card" style="animation-delay: 0.1s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <rect x="3" y="3" width="18" height="18" rx="2" stroke="white" stroke-width="2"/>
                            <path d="M3 9h18M9 3v18" stroke="white" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="spec-label">Model Type</div>
                    <div class="spec-value">${specs.type}</div>
                </div>

                <div class="spec-card" style="animation-delay: 0.2s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <circle cx="12" cy="12" r="9" stroke="white" stroke-width="2"/>
                            <path d="M12 7v5l3 3" stroke="white" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div class="spec-label">Parameters</div>
                    <div class="spec-value">${formatNumber(specs.parameters)}</div>
                </div>

                <div class="spec-card" style="animation-delay: 0.3s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <rect x="4" y="4" width="16" height="16" rx="2" stroke="white" stroke-width="2"/>
                            <path d="M4 10h16M10 4v16" stroke="white" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="spec-label">RAM Required</div>
                    <div class="spec-value">${specs.ram}<span class="spec-unit">GB</span></div>
                </div>

                <div class="spec-card" style="animation-delay: 0.4s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" stroke="white" stroke-width="2" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="spec-label">VRAM Required</div>
                    <div class="spec-value">${specs.vram}<span class="spec-unit">GB</span></div>
                </div>

                <div class="spec-card" style="animation-delay: 0.5s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="white" stroke-width="2" stroke-linejoin="round"/>
                            <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="white" stroke-width="2" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="spec-label">Precision</div>
                    <div class="spec-value">${specs.precision}</div>
                </div>

                <div class="spec-card" style="animation-delay: 0.6s;">
                    <div class="spec-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M20 7h-4V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2H4a2 2 0 00-2 2v10a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2z" stroke="white" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="spec-label">Organization</div>
                    <div class="spec-value" style="font-size: 18px;">${model.author || 'Unknown'}</div>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Error fetching model details:', error);
        specsContainer.innerHTML = `
            <div class="empty-state">
                <h3>Error loading model details</h3>
                <p>Please try selecting another model</p>
            </div>
        `;
    }
}

// Format large numbers
function formatNumber(num) {
    if (typeof num === 'string') return num;
    if (num >= 1e12) return `${(num / 1e12).toFixed(1)}T`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
}

// Show empty state
function showEmptyState() {
    specsContainer.innerHTML = `
        <div class="empty-state">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                <circle cx="40" cy="40" r="38" stroke="url(#emptyGradient)" stroke-width="2" opacity="0.3"/>
                <path d="M40 20v20m0 0v20m0-20h20m-20 0H20" stroke="url(#emptyGradient)" stroke-width="2" stroke-linecap="round" opacity="0.3"/>
                <defs>
                    <linearGradient id="emptyGradient" x1="0" y1="0" x2="80" y2="80">
                        <stop offset="0%" stop-color="#667eea"/>
                        <stop offset="100%" stop-color="#764ba2"/>
                    </linearGradient>
                </defs>
            </svg>
            <h3>Select a model to begin</h3>
            <p>Choose from thousands of AI models to see their specifications</p>
        </div>
    `;
    hideActionSection();
    hideResultsSection();
}

// Show action section
function showActionSection() {
    actionSection.style.display = 'block';
    actionSection.style.animation = 'fadeInUp 0.6s ease-out';
}

// Hide action section
function hideActionSection() {
    actionSection.style.display = 'none';
}

// Show results section
function showResultsSection() {
    resultsSection.style.display = 'block';
    resultsSection.style.animation = 'fadeInUp 0.6s ease-out';

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Hide results section
function hideResultsSection() {
    resultsSection.style.display = 'none';
}

// Find hardware for selected model
async function findHardware(model) {
    try {
        // Update button to loading state
        findHardwareBtn.disabled = true;
        findHardwareBtn.classList.add('loading');
        findHardwareBtn.innerHTML = `
            <span class="button-content">
                <svg class="button-icon" width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2"/>
                    <path d="M10 3v4M10 13v4M17 10h-4M7 10H3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                Researching Hardware...
            </span>
            <div class="button-shimmer"></div>
        `;

        // Show results section with loading state
        showResultsSection();
        resultsContent.innerHTML = `
            <div style="text-align: center; padding: 40px 20px;">
                <div class="loading-dot" style="display: inline-block; width: 12px; height: 12px; margin-bottom: 20px;"></div>
                <p style="color: var(--text-secondary); font-size: 16px;">Claude is researching the best hardware options for your model...</p>
            </div>
        `;

        // Make API request
        const response = await fetch('/api/find-hardware', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                modelId: model.id,
                modelName: model.modelId || model.id,
                author: model.author
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Display results
        displayResults(data.recommendation);

    } catch (error) {
        console.error('Error finding hardware:', error);
        resultsContent.innerHTML = `
            <div style="text-align: center; padding: 40px 20px;">
                <h3 style="color: #f5576c; margin-bottom: 12px;">Error</h3>
                <p style="color: var(--text-secondary);">${error.message || 'Failed to get hardware recommendations. Please try again.'}</p>
            </div>
        `;
    } finally {
        // Reset button state
        findHardwareBtn.disabled = false;
        findHardwareBtn.classList.remove('loading');
        findHardwareBtn.innerHTML = `
            <span class="button-content">
                <svg class="button-icon" width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M10 1v18M19 10H1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2" opacity="0.3"/>
                </svg>
                Find Perfect Hardware
            </span>
            <div class="button-shimmer"></div>
        `;
    }
}

// Display hardware recommendations
function displayResults(recommendation) {
    // Convert markdown-style formatting to HTML
    let html = recommendation
        .replace(/### (.*)/g, '<h3>$1</h3>')
        .replace(/## (.*)/g, '<h3>$1</h3>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/^- (.*)/gm, '<li>$1</li>');

    // Wrap list items in ul tags
    html = html.replace(/(<li>.*<\/li>\s*)+/gs, '<ul>$&</ul>');

    // Wrap in paragraphs if not already wrapped
    if (!html.startsWith('<h3>') && !html.startsWith('<p>')) {
        html = '<p>' + html;
    }
    if (!html.endsWith('</p>') && !html.endsWith('</ul>')) {
        html = html + '</p>';
    }

    resultsContent.innerHTML = html;

    // Trigger success celebration
    triggerSuccessCelebration();
}

// Success celebration with confetti
function triggerSuccessCelebration() {
    const overlay = document.createElement('div');
    overlay.className = 'success-overlay';
    document.body.appendChild(overlay);

    const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];

    // Create confetti particles
    for (let i = 0; i < 50; i++) {
        setTimeout(() => {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.animationDuration = (2 + Math.random() * 2) + 's';
            confetti.style.animationDelay = (Math.random() * 0.5) + 's';

            overlay.appendChild(confetti);

            // Remove confetti after animation
            setTimeout(() => confetti.remove(), 3500);
        }, i * 30);
    }

    // Remove overlay after all confetti is done
    setTimeout(() => overlay.remove(), 4000);
}

// Initialize the app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
