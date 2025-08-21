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
        results.classList.add('hidden');
        errorDisplay.classList.add('hidden');
        successResults.classList.add('hidden');
    }

    function showError(message) {
        hideResults();
        errorMessage.textContent = message;
        errorDisplay.classList.remove('hidden');
        results.classList.remove('hidden');
        
        // Scroll to error
        errorDisplay.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function displayResults(data) {
        hideResults();
        
        // Populate content
        document.getElementById('answerContent').innerHTML = formatText(data.answer);
        document.getElementById('reasoningContent').innerHTML = formatText(data.reasoning);
        document.getElementById('evidenceContent').innerHTML = formatText(data.evidence);
        
        // Populate statistics
        const tokenStats = data.token_stats || {};
        const timingStats = data.timing_stats || {};
        const chunkStats = data.chunk_stats || {};
        
        document.getElementById('tokenStats').textContent = formatTokenStats(tokenStats);
        document.getElementById('timingStats').textContent = formatTimingStats(tokenStats); // Use tokenStats for timing
        document.getElementById('chunkStats').textContent = formatChunkStats(tokenStats); // Use tokenStats for chunk stats
        
        // Show results
        successResults.classList.remove('hidden');
        results.classList.remove('hidden');
        
        // Add animation class
        successResults.classList.add('result-panel');
        
        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
        // Check if we have the new detailed format with timing breakdown
        if (stats.timing) {
            const timing = stats.timing;
            const mapTime = timing.map_phase_time || 0;
            const reduceTime = timing.reduce_phase_time || 0;
            const totalTime = timing.total_time || 0;
            
            return `${totalTime.toFixed(1)}s total (${mapTime.toFixed(1)}s map, ${reduceTime.toFixed(1)}s reduce)`;
        }
        
        // Fallback to old format
        const total = stats.total_time || 0;
        return `${total.toFixed(1)}s`;
    }

    function formatChunkStats(stats) {
        // Check if we have the new detailed format with filtering stats
        if (stats.filtering_stats) {
            const filtering = stats.filtering_stats;
            const totalDocs = stats.len_docs || 0;
            const totalChunks = filtering.total_chunks || 0;
            const afterFiltering = filtering.chunks_after_filtering || 0;
            
            let result = `${totalChunks} chunks`;
            if (totalDocs > 0) {
                result += ` (${totalDocs} docs)`;
            }
            if (afterFiltering !== totalChunks) {
                result += `, ${afterFiltering} after filtering`;
            }
            return result;
        }
        
        // Fallback to old format
        const total = stats.total_chunks || 0;
        const processed = stats.chunks_after_filtering || 0;
        
        if (total === processed) {
            return `${total} chunks`;
        } else {
            return `${processed}/${total} chunks`;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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