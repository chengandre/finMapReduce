// Enhanced MapReduce QA WebApp JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('qaForm');
    const fileInput = document.getElementById('file');
    const dropZone = document.getElementById('dropZone');
    const dropText = document.getElementById('dropText');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileStats = document.getElementById('fileStats');
    const previewBtn = document.getElementById('previewBtn');
    const questionInput = document.getElementById('question');
    const exampleBtn = document.getElementById('exampleBtn');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedOptions = document.getElementById('advancedOptions');
    const chevron = document.getElementById('chevron');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Pipeline type elements
    const pipelineTypeSelect = document.getElementById('pipeline_type');
    const formatTypeContainer = document.getElementById('format_type_container');
    const strategyContainer = document.getElementById('strategy_container');
    const truncationOptions = document.getElementById('truncationOptions');

    // Results elements
    const welcomeMessage = document.getElementById('welcomeMessage');
    const errorDisplay = document.getElementById('errorDisplay');
    const successResults = document.getElementById('successResults');
    const errorMessage = document.getElementById('errorMessage');
    const resultsHeader = document.getElementById('resultsHeader');

    // Preview elements (using correct IDs from enhanced HTML)
    const documentPreviewPanel = document.getElementById('documentPreviewPanel');
    const defaultPreviewMessage = document.getElementById('defaultPreviewMessage');
    const actualDocumentPreview = document.getElementById('actualDocumentPreview');
    const previewLoadingSpinner = document.getElementById('previewLoadingSpinner');
    const previewContent = document.getElementById('previewContent');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const previewFileType = document.getElementById('previewFileType');
    const previewFileModified = document.getElementById('previewFileModified');

    // Example questions for different domains
    const exampleQuestions = {
        financial: [
            "What are the key financial highlights mentioned in this document?",
            "What were the main revenue sources and their performance?",
            "What risks or challenges are identified in this document?",
            "What are the company's future plans or outlook?",
            "What were the operating expenses and how did they change?"
        ],
        general: [
            "What are the main points discussed in this document?",
            "Can you summarize the key findings or conclusions?",
            "What recommendations or next steps are mentioned?",
            "What problems or challenges are identified?",
            "What data or statistics are presented?"
        ]
    };

    // Initialize the interface
    init();

    function init() {
        setupEventListeners();
        updatePipelineOptions();
        loadAvailableOptions();
    }

    function setupEventListeners() {
        // File upload handling
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', handleFileInputChange);

        // Form submission
        form.addEventListener('submit', handleFormSubmit);

        // UI interactions
        exampleBtn.addEventListener('click', insertExampleQuestion);
        advancedToggle.addEventListener('click', toggleAdvancedOptions);
        pipelineTypeSelect.addEventListener('change', updatePipelineOptions);

        // Provider model mapping
        document.getElementById('provider').addEventListener('change', updateModelOptions);

        // Preview functionality
        if (previewBtn) {
            previewBtn.addEventListener('click', showDocumentPreview);
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        dropZone.classList.add('border-blue-400', 'bg-blue-50');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        dropZone.classList.remove('border-blue-400', 'bg-blue-50');
    }

    function handleDrop(e) {
        e.preventDefault();
        dropZone.classList.remove('border-blue-400', 'bg-blue-50');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    }

    function handleFileInputChange(e) {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    }

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
        fileStats.textContent = `${(file.size / 1024).toFixed(1)} KB • ${file.type || getFileTypeFromExtension(file.name)}`;

        // Set the file input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        // Hide any previous errors
        hideError();

        // Show document preview in the second panel
        showDocumentPreviewInPanel(file);
    }

    function insertExampleQuestion() {
        const allQuestions = [...exampleQuestions.financial, ...exampleQuestions.general];
        const randomQuestion = allQuestions[Math.floor(Math.random() * allQuestions.length)];
        questionInput.value = randomQuestion;
    }

    function toggleAdvancedOptions() {
        const isHidden = advancedOptions.classList.contains('hidden');

        if (isHidden) {
            advancedOptions.classList.remove('hidden');
            chevron.style.transform = 'rotate(90deg)';
        } else {
            advancedOptions.classList.add('hidden');
            chevron.style.transform = 'rotate(0deg)';
        }
    }

    function updatePipelineOptions() {
        const pipelineType = pipelineTypeSelect.value;

        if (pipelineType === 'mapreduce') {
            formatTypeContainer.classList.remove('hidden');
            strategyContainer.classList.add('hidden');
            truncationOptions.classList.add('hidden');
        } else {
            formatTypeContainer.classList.add('hidden');
            strategyContainer.classList.remove('hidden');
            truncationOptions.classList.remove('hidden');
        }
    }

    function updateModelOptions() {
        const provider = document.getElementById('provider').value;
        const modelSelect = document.getElementById('model_name');

        // Clear existing options
        modelSelect.innerHTML = '';

        const modelsByProvider = {
            'openai': [
                { value: 'gpt-4o-mini', text: 'GPT-4o Mini' },
                { value: 'gpt-4o', text: 'GPT-4o' },
                { value: 'gpt-4-turbo', text: 'GPT-4 Turbo' },
                { value: 'gpt-3.5-turbo', text: 'GPT-3.5 Turbo' }
            ],
            'openrouter': [
                { value: 'deepseek/deepseek-r1-0528:free', text: 'DeepSeek R1 (Free)' },
                { value: 'anthropic/claude-3-haiku', text: 'Claude 3 Haiku' },
                { value: 'meta-llama/llama-3.1-8b-instruct:free', text: 'Llama 3.1 8B (Free)' }
            ]
        };

        const models = modelsByProvider[provider] || modelsByProvider['openai'];
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            modelSelect.appendChild(option);
        });
    }

    async function loadAvailableOptions() {
        try {
            const response = await fetch('/api/models');
            const options = await response.json();

            // Update prompt sets if available
            if (options.prompt_sets) {
                const promptSelect = document.getElementById('prompt_set');
                promptSelect.innerHTML = '';
                options.prompt_sets.forEach(set => {
                    const option = document.createElement('option');
                    option.value = set;
                    option.textContent = set.charAt(0).toUpperCase() + set.slice(1);
                    if (set === 'hybrid') option.selected = true;
                    promptSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.warn('Could not load available options:', error);
        }
    }

    async function handleFormSubmit(e) {
        e.preventDefault();

        // Validation
        if (!fileInput.files[0]) {
            showError('Please upload a document first.');
            return;
        }

        if (!questionInput.value.trim()) {
            showError('Please enter a question.');
            return;
        }

        // Show loading state
        setLoading(true);
        hideError();
        hideResults();

        try {
            // Prepare form data
            const formData = new FormData();

            // Add all form fields
            const formElements = form.elements;
            for (let element of formElements) {
                if (element.name && element.value && element.type !== 'submit') {
                    if (element.type === 'file') {
                        formData.append(element.name, element.files[0]);
                    } else {
                        formData.append(element.name, element.value);
                    }
                }
            }

            // Add pipeline-specific parameters
            const pipelineType = pipelineTypeSelect.value;
            if (pipelineType === 'truncation') {
                // For truncation, format_type is not used
                formData.delete('format_type');
            }

            // Submit request
            const response = await fetch('/api/answer', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error?.message || 'Request failed');
            }

            // Display results
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred while processing your request.');
        } finally {
            setLoading(false);
        }
    }

    function displayResults(data) {
        // Hide welcome message
        if (welcomeMessage) {
            welcomeMessage.classList.add('hidden');
        }

        // Update header
        if (resultsHeader) {
            const h2 = resultsHeader.querySelector('h2');
            const p = resultsHeader.querySelector('p');
            if (h2) h2.textContent = 'Analysis Complete';
            if (p) p.textContent = 'Results for your document analysis';
        }

        // Display answer (with markdown rendering)
        const answerElement = document.getElementById('answerContent');
        if (answerElement) {
            answerElement.innerHTML = renderMarkdown(data.answer || 'No answer provided');
        }

        // Display reasoning (with markdown rendering)
        const reasoningElement = document.getElementById('reasoningContent');
        if (reasoningElement) {
            reasoningElement.innerHTML = renderMarkdown(data.reasoning || 'No reasoning provided');
        }

        // Display evidence (handle array or string)
        const evidenceElement = document.getElementById('evidenceContent');
        if (evidenceElement) {
            if (Array.isArray(data.evidence)) {
                evidenceElement.innerHTML = data.evidence.map(item => renderMarkdown(item)).join('<hr class="my-3">');
            } else {
                evidenceElement.innerHTML = renderMarkdown(data.evidence || 'No evidence provided');
            }
        }

        // Display statistics
        updateStatistics(data);

        // Show results
        successResults.classList.remove('hidden');
    }

    function updateStatistics(data) {
        const { token_stats, timing_stats, chunk_stats } = data;

        // Token stats
        const tokenElement = document.getElementById('tokenStats');
        if (token_stats && token_stats.total) {
            const total = token_stats.total;
            const input = total.input_tokens || 0;
            const output = total.output_tokens || 0;
            const cache = total.cache_read_tokens || 0;
            const totalTokens = input + output;

            let tokenText = `${totalTokens.toLocaleString()} total`;
            if (input > 0 || output > 0) {
                tokenText += ` (${input.toLocaleString()} in, ${output.toLocaleString()} out)`;
            }
            if (cache > 0) {
                tokenText += `, ${cache.toLocaleString()} cached`;
            }
            tokenElement.textContent = tokenText;
        } else {
            tokenElement.textContent = 'N/A';
        }

        // Timing stats
        const timingElement = document.getElementById('timingStats');
        if (timing_stats && Object.keys(timing_stats).length > 0) {
            const mapTime = timing_stats.map_phase_time || 0;
            const reduceTime = timing_stats.reduce_phase_time || 0;
            const llmTime = timing_stats.llm_call_time || 0;
            const totalTime = timing_stats.total_time || 0;

            if (mapTime > 0 && reduceTime > 0) {
                // MapReduce pipeline timing
                timingElement.textContent = `${totalTime.toFixed(1)}s total (${mapTime.toFixed(1)}s map, ${reduceTime.toFixed(1)}s reduce)`;
            } else if (llmTime > 0) {
                // Truncation pipeline timing
                const truncationTime = timing_stats.truncation_time || 0;
                timingElement.textContent = `${totalTime.toFixed(1)}s total (${llmTime.toFixed(1)}s LLM, ${truncationTime.toFixed(1)}s prep)`;
            } else if (totalTime > 0) {
                // Basic timing
                timingElement.textContent = `${totalTime.toFixed(1)}s`;
            } else {
                timingElement.textContent = 'N/A';
            }
        } else {
            timingElement.textContent = 'N/A';
        }

        // Chunk stats
        const chunkElement = document.getElementById('chunkStats');
        if (chunk_stats && Object.keys(chunk_stats).length > 0) {
            const totalDocs = chunk_stats.len_docs || 0;
            const filtering = chunk_stats.filtering_stats || {};
            const totalChunks = filtering.chunks_before_filtering || chunk_stats.total_chunks || 0;
            const afterFiltering = filtering.chunks_after_filtering || 0;

            if (totalChunks > 0) {
                let chunkText = `${totalChunks} chunks`;
                if (totalDocs > 0) {
                    chunkText += ` (${totalDocs} docs)`;
                }
                if (afterFiltering > 0 && afterFiltering !== totalChunks) {
                    chunkText += `, ${afterFiltering} after filtering`;
                }
                chunkElement.textContent = chunkText;
            } else {
                chunkElement.textContent = 'N/A';
            }
        } else {
            chunkElement.textContent = 'N/A';
        }

        // Pipeline type
        const pipelineElement = document.getElementById('pipelineStats');
        const pipelineType = pipelineTypeSelect.value;
        const formatType = document.getElementById('format_type').value;
        if (pipelineType === 'mapreduce') {
            pipelineElement.textContent = `MapReduce (${formatType})`;
        } else {
            const strategy = document.getElementById('strategy').value;
            pipelineElement.textContent = `Truncation (${strategy})`;
        }
    }

    function renderMarkdown(text) {
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        } else {
            // Fallback: simple text with line breaks
            return text.replace(/\n/g, '<br>');
        }
    }

    function setLoading(loading) {
        if (loading) {
            submitBtn.disabled = true;
            submitText.textContent = 'Processing...';
            loadingSpinner.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            submitText.textContent = 'Analyze Document';
            loadingSpinner.classList.add('hidden');
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorDisplay.classList.remove('hidden');
        hideResults();
    }

    function hideError() {
        errorDisplay.classList.add('hidden');
    }

    function hideResults() {
        successResults.classList.add('hidden');
        welcomeMessage.classList.remove('hidden');

        // Reset header
        resultsHeader.querySelector('h2').textContent = 'Analysis Results';
        resultsHeader.querySelector('p').textContent = 'Upload a document and ask a question to see results here';
    }

    function getFileTypeFromExtension(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const types = {
            'pdf': 'PDF Document',
            'txt': 'Text File',
            'md': 'Markdown File'
        };
        return types[ext] || 'Unknown';
    }

    async function showDocumentPreview() {
        // This function is no longer used in the enhanced version
        // Preview is handled automatically in showDocumentPreviewInPanel
    }

    function hideDocumentPreview() {
        // This function is no longer used in the enhanced version
        // Show welcome message if no results are shown
        if (successResults && successResults.classList.contains('hidden')) {
            if (welcomeMessage) {
                welcomeMessage.classList.remove('hidden');
            }
        }
    }

    // Function to show document preview in the second panel (new 3-panel layout)
    function showDocumentPreviewInPanel(file) {
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        // Get new preview panel elements
        const defaultPreviewMessage = document.getElementById('defaultPreviewMessage');
        const actualDocumentPreview = document.getElementById('actualDocumentPreview');
        const previewLoadingSpinner = document.getElementById('previewLoadingSpinner');
        const previewContent = document.getElementById('previewContent');
        const previewFileName = document.getElementById('previewFileName');
        const previewFileSize = document.getElementById('previewFileSize');
        const previewFileType = document.getElementById('previewFileType');
        const previewFileModified = document.getElementById('previewFileModified');

        // Hide default message and show actual preview
        if (defaultPreviewMessage) defaultPreviewMessage.classList.add('hidden');
        if (actualDocumentPreview) actualDocumentPreview.classList.remove('hidden');

        // Update file info
        if (previewFileName) previewFileName.textContent = file.name;
        if (previewFileSize) previewFileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        if (previewFileType) previewFileType.textContent = fileExt.toUpperCase().replace('.', '');
        if (previewFileModified) previewFileModified.textContent = new Date(file.lastModified).toLocaleString();

        if (fileExt === '.txt' || fileExt === '.md') {
            // For text files, show content preview
            if (previewLoadingSpinner) previewLoadingSpinner.classList.remove('hidden');
            if (previewContent) previewContent.textContent = 'Loading...';

            const reader = new FileReader();
            reader.onload = function(e) {
                const content = e.target.result;
                const previewText = content.length > 8000 ?
                    content.substring(0, 8000) + '\n\n... (file truncated for preview, full content will be processed)' :
                    content;

                if (previewLoadingSpinner) previewLoadingSpinner.classList.add('hidden');
                if (previewContent) {
                    if (fileExt === '.md') {
                        // Render markdown as HTML
                        previewContent.innerHTML = renderMarkdown(previewText);
                        previewContent.classList.add('markdown-content');
                    } else {
                        // Display as plain text
                        previewContent.textContent = previewText;
                        previewContent.classList.remove('markdown-content');
                    }
                }
            };
            reader.readAsText(file);
        } else if (fileExt === '.pdf') {
            // For PDFs, show placeholder
            if (previewLoadingSpinner) previewLoadingSpinner.classList.add('hidden');
            if (previewContent) {
                previewContent.innerHTML = `
                    <div class="flex items-center justify-center h-32 text-gray-400 border-2 border-dashed border-gray-200 rounded">
                        <div class="text-center">
                            <svg class="mx-auto h-12 w-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/>
                            </svg>
                            <p class="text-sm font-medium">PDF Document</p>
                            <p class="text-xs text-gray-500 mt-1">Preview will be available after processing</p>
                        </div>
                    </div>
                `;
            }
        }
    }
});