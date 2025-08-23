// MapReduce QA WebApp JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('qaForm');
    const fileInput = document.getElementById('file');
    const dropZone = document.getElementById('dropZone');
    const dropText = document.getElementById('dropText');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const questionInput = document.getElementById('question');
    const exampleBtn = document.getElementById('exampleBtn');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedOptions = document.getElementById('advancedOptions');
    const chevron = document.getElementById('chevron');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const results = document.getElementById('results');
    const errorDisplay = document.getElementById('errorDisplay');
    const successResults = document.getElementById('successResults');
    const errorMessage = document.getElementById('errorMessage');
    const documentPreview = document.getElementById('documentPreview');

    // Example questions
    const exampleQuestions = [
        "What are the key financial highlights mentioned in this document?",
        "What were the main revenue sources and their performance?",
        "What risks or challenges are identified in this document?",
        "What are the company's future plans or outlook?",
        "What were the operating expenses and how did they change?"
    ];

    // File upload handling
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        // Validate file type
        const allowedTypes = ['.pdf', '.txt', '.md'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExt)) {
            showError('Unsupported file type. Please upload a PDF, TXT, or MD file.');
            return;
        }
        
        // Validate file size (50MB)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (50MB).`);
            return;
        }
        
        // Update UI
        dropText.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        fileName.textContent = file.name;
        
        // Set the file input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        
        // Show document preview
        showDocumentPreview(file);
    }

    // Example question button
    exampleBtn.addEventListener('click', () => {
        const randomQuestion = exampleQuestions[Math.floor(Math.random() * exampleQuestions.length)];
        questionInput.value = randomQuestion;
    });

    // Advanced options toggle
    advancedToggle.addEventListener('click', () => {
        const isHidden = advancedOptions.classList.contains('hidden');
        
        if (isHidden) {
            advancedOptions.classList.remove('hidden');
            chevron.style.transform = 'rotate(90deg)';
        } else {
            advancedOptions.classList.add('hidden');
            chevron.style.transform = 'rotate(0deg)';
        }
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results
        hideResults();
        
        // Show loading state
        setLoadingState(true);
        
        try {
            // Prepare form data
            const formData = new FormData(form);
            
            // Make API request
            const response = await fetch('/api/answer', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error?.message || 'Server error');
            }
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An unexpected error occurred');
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(loading) {
        if (loading) {
            submitBtn.disabled = true;
            submitText.textContent = 'Processing...';
            loadingSpinner.classList.remove('hidden');
            submitBtn.classList.add('loading');
        } else {
            submitBtn.disabled = false;
            submitText.textContent = 'Analyze Document';
            loadingSpinner.classList.add('hidden');
            submitBtn.classList.remove('loading');
        }
    }

    function hideResults() {
        results.innerHTML = `
            <div class="flex items-center justify-center h-64 text-gray-400">
                <div class="text-center">
                    <svg class="mx-auto h-12 w-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    <p>Results will appear here after analysis</p>
                </div>
            </div>
        `;
    }

    function showError(message) {
        results.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-md p-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">Error</h3>
                        <div class="mt-2 text-sm text-red-700">
                            <p>${escapeHtml(message)}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function displayResults(data) {
        const tokenStats = data.token_stats || {};
        const timingStats = data.timing_stats || {};
        const chunkStats = data.chunk_stats || {};
        
        results.innerHTML = `
            <div class="space-y-4">
                <!-- Answer -->
                <div class="result-content">
                    <h3>
                        <svg class="text-green-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                        Answer
                    </h3>
                    <div class="text-sm text-gray-700">${formatText(data.answer)}</div>
                </div>
                
                <!-- Reasoning -->
                <div class="result-content">
                    <h3>
                        <svg class="text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
                        </svg>
                        Reasoning
                    </h3>
                    <div class="text-sm text-gray-700">${formatText(data.reasoning)}</div>
                </div>
                
                <!-- Evidence -->
                <div class="result-content">
                    <h3>
                        <svg class="text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm0 4a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1V8zm8 0a1 1 0 011-1h4a1 1 0 011 1v2a1 1 0 01-1 1h-4a1 1 0 01-1-1V8zm0 4a1 1 0 011-1h4a1 1 0 011 1v2a1 1 0 01-1 1h-4a1 1 0 01-1-1v-2z" clip-rule="evenodd"/>
                        </svg>
                        Evidence
                    </h3>
                    <div class="text-sm text-gray-700">${formatText(data.evidence)}</div>
                </div>
                
                <!-- Statistics -->
                <div class="result-content">
                    <h3>
                        <svg class="text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
                        </svg>
                        Processing Statistics
                    </h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Tokens Used</div>
                            <div class="stat-value">${formatTokenStats(tokenStats)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Processing Time</div>
                            <div class="stat-value">${formatTimingStats(timingStats)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Document Chunks</div>
                            <div class="stat-value">${formatChunkStats(chunkStats)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Pipeline Type</div>
                            <div class="stat-value">${determinePipelineType(tokenStats, chunkStats)}</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add animation
        results.classList.add('result-panel');
    }

    function formatText(text) {
        if (!text) return '<p class="text-gray-500 italic">No content available</p>';
        
        // Handle arrays (like evidence field)
        if (Array.isArray(text)) {
            if (text.length === 0) return '<p class="text-gray-500 italic">No content available</p>';
            return text
                .map(item => `<p>${escapeHtml(String(item))}</p>`)
                .join('');
        }
        
        // Handle strings
        return String(text)
            .split('\n\n')
            .map(paragraph => paragraph.trim())
            .filter(paragraph => paragraph.length > 0)
            .map(paragraph => `<p>${escapeHtml(paragraph)}</p>`)
            .join('');
    }

    function formatTokenStats(stats) {
        // Handle the new detailed format
        if (stats.total) {
            const total = stats.total;
            const input = total.input_tokens || 0;
            const output = total.output_tokens || 0;
            const cache = total.cache_read_tokens || 0;
            const totalTokens = input + output;
            
            let result = `${totalTokens.toLocaleString()} total (${input.toLocaleString()} in, ${output.toLocaleString()} out)`;
            if (cache > 0) {
                result += `, ${cache.toLocaleString()} cached`;
            }
            return result;
        }
        
        // Fallback to old format
        const input = stats.input_tokens || 0;
        const output = stats.output_tokens || 0;
        const total = input + output;
        
        return `${total.toLocaleString()} total (${input.toLocaleString()} in, ${output.toLocaleString()} out)`;
    }

    function formatTimingStats(stats) {
        // Check if we have timing data
        if (!stats || Object.keys(stats).length === 0) {
            return 'N/A';
        }
        
        // Check if we have the detailed format with timing breakdown
        const mapTime = stats.map_phase_time || 0;
        const reduceTime = stats.reduce_phase_time || 0;
        const llmTime = stats.llm_call_time || 0;
        const totalTime = stats.total_time || 0;
        
        if (mapTime > 0 && reduceTime > 0) {
            // MapReduce pipeline timing
            return `${totalTime.toFixed(1)}s total (${mapTime.toFixed(1)}s map, ${reduceTime.toFixed(1)}s reduce)`;
        } else if (llmTime > 0) {
            // Truncation pipeline timing
            const truncationTime = stats.truncation_time || 0;
            return `${totalTime.toFixed(1)}s total (${llmTime.toFixed(1)}s LLM, ${truncationTime.toFixed(1)}s prep)`;
        } else if (totalTime > 0) {
            // Basic timing
            return `${totalTime.toFixed(1)}s`;
        }
        
        return 'N/A';
    }

    function formatChunkStats(stats) {
        // Check if we have chunk data
        if (!stats || Object.keys(stats).length === 0) {
            return 'N/A';
        }
        
        // Check if we have the detailed format with filtering stats
        if (stats.filtering_stats) {
            const filtering = stats.filtering_stats;
            const totalDocs = stats.len_docs || 0;
            const totalChunks = filtering.chunks_before_filtering || stats.total_chunks || 0;
            const afterFiltering = filtering.chunks_after_filtering || 0;
            
            let result = `${totalChunks} chunks`;
            if (totalDocs > 0) {
                result += ` (${totalDocs} docs)`;
            }
            if (afterFiltering !== totalChunks && afterFiltering > 0) {
                result += `, ${afterFiltering} after filtering`;
            }
            return result;
        }
        
        // Check for simple total chunks
        const totalChunks = stats.total_chunks || 0;
        const afterFiltering = stats.chunks_after_filtering || 0;
        const totalDocs = stats.len_docs || 0;
        
        if (totalChunks > 0) {
            let result = `${totalChunks} chunks`;
            if (totalDocs > 0) {
                result += ` (${totalDocs} docs)`;
            }
            if (afterFiltering > 0 && afterFiltering !== totalChunks) {
                result += `, ${afterFiltering} after filtering`;
            }
            return result;
        }
        
        return 'N/A';
    }

    function determinePipelineType(tokenStats, chunkStats) {
        // Check for MapReduce indicators
        if (tokenStats.map_phase && tokenStats.reduce_phase) {
            // Check if it's hybrid format
            if (chunkStats.filtering_stats) {
                return 'MapReduce (hybrid)';
            }
            return 'MapReduce';
        }
        
        // Check for truncation indicators
        if (tokenStats.truncation_stats || tokenStats.llm_call) {
            return 'Truncation';
        }
        
        return 'Unknown';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function showDocumentPreview(file) {
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        // Show file info
        const fileInfo = `
            <div class="document-info">
                <div><strong>File:</strong> ${escapeHtml(file.name)}</div>
                <div><strong>Type:</strong> ${fileExt.toUpperCase()}</div>
                <div><strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB</div>
                <div><strong>Modified:</strong> ${new Date(file.lastModified).toLocaleString()}</div>
            </div>
        `;
        
        if (fileExt === '.txt' || fileExt === '.md') {
            // For text files, show content preview
            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                const previewContent = content.length > 5000 ? 
                    content.substring(0, 5000) + '\n\n... (truncated, full content will be processed)' : 
                    content;
                
                documentPreview.innerHTML = fileInfo + 
                    `<div class="text-xs text-gray-500 mb-2">Preview (${fileExt === '.md' ? 'Markdown' : 'Text'}):</div>` +
                    `<pre class="whitespace-pre-wrap text-xs">${escapeHtml(previewContent)}</pre>`;
                documentPreview.classList.add('has-content');
            };
            reader.readAsText(file);
        } else if (fileExt === '.pdf') {
            // For PDFs, show file info only
            documentPreview.innerHTML = fileInfo + 
                `<div class="flex items-center justify-center h-32 text-gray-400 border-2 border-dashed border-gray-200 rounded">
                    <div class="text-center">
                        <svg class="mx-auto h-8 w-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
                        </svg>
                        <p class="text-xs">PDF Preview Not Available</p>
                        <p class="text-xs">File will be processed during analysis</p>
                    </div>
                </div>`;
            documentPreview.classList.add('has-content');
        }
    }

    // Provider/model relationship handling
    const providerSelect = document.getElementById('provider');
    const modelSelect = document.getElementById('model_name');
    
    providerSelect.addEventListener('change', (e) => {
        updateModelOptions(e.target.value);
    });
    
    function updateModelOptions(provider) {
        // Clear existing options
        modelSelect.innerHTML = '';
        
        let models = [];
        if (provider === 'openai') {
            models = [
                { value: 'gpt-4o-mini', text: 'GPT-4o Mini' },
                { value: 'gpt-4o', text: 'GPT-4o' },
                { value: 'gpt-4-turbo', text: 'GPT-4 Turbo' },
                { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo' }
            ];
        } else if (provider === 'openrouter') {
            models = [
                { value: 'openai/gpt-4o-mini', text: 'GPT-4o Mini' },
                { value: 'openai/gpt-4o', text: 'GPT-4o' },
                { value: 'anthropic/claude-3.5-sonnet', text: 'Claude 3.5 Sonnet' },
                { value: 'deepseek/deepseek-r1-0528:free', text: 'DeepSeek R1 (Free)' }
            ];
        }
        
        // Add model options
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            modelSelect.appendChild(option);
        });
        
        // Set default selection
        if (models.length > 0) {
            modelSelect.value = models[0].value;
        }
    }
    
    // Initialize model options
    updateModelOptions(providerSelect.value);
    
    // Initialize results panel
    hideResults();
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter or Cmd+Enter to submit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (!submitBtn.disabled && fileInput.files.length > 0 && questionInput.value.trim()) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to hide results
        if (e.key === 'Escape') {
            hideResults();
        }
    });
});