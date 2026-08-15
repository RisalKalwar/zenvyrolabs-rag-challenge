const fileInput = document.getElementById('pdf-upload');
const uploadStatus = document.getElementById('upload-status');
const uploadSuccess = document.getElementById('upload-success');
const successText = document.getElementById('success-text');

const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatBox = document.getElementById('chat-box');

// Base API URL - assumes FastAPI is running on the same host:port
// If the backend runs on port 8000 while frontend is opened locally, this needs adjustment.
// But since FastAPI serves this frontend statically, they share the origin.
const API_BASE = window.location.origin;

// Track the current book mode
let currentMode = 'coding';

const defaultWelcome = `
    <div class="message system-msg">
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-bubble">
            <p>Welcome! Please upload a PDF document on the left, and then ask me anything about it.</p>
        </div>
    </div>
`;

// BUG 3 FIX: each mode now also tracks its own chatHistory array, so
// switching modes keeps separate conversation memory per book, and each
// entry is sent to the backend on every request instead of just the
// latest message.
const modeStates = {
    coding: { chatHtml: defaultWelcome, successMsg: '', chatHistory: [] },
    novel: { chatHtml: defaultWelcome, successMsg: '', chatHistory: [] },
    manga: { chatHtml: defaultWelcome, successMsg: '', chatHistory: [] }
};

// Function to switch modes
window.setMode = function(mode) {
    // 1. Save current state before switching
    modeStates[currentMode].chatHtml = chatBox.innerHTML;
    modeStates[currentMode].successMsg = uploadSuccess.classList.contains('hidden') ? '' : successText.innerText;

    // 2. Switch mode visually
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');

    // 3. Restore state for the new mode
    chatBox.innerHTML = modeStates[currentMode].chatHtml;
    
    if (modeStates[currentMode].successMsg) {
        successText.innerText = modeStates[currentMode].successMsg;
        uploadSuccess.classList.remove('hidden');
    } else {
        uploadSuccess.classList.add('hidden');
    }
    
    chatBox.scrollTop = chatBox.scrollHeight;
};

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading state
    uploadStatus.classList.remove('hidden');
    uploadSuccess.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('book_type', currentMode);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        uploadStatus.classList.add('hidden');
        if (data.error) {
            alert(data.error);
        } else {
            successText.innerText = data.message || `Successfully embedded as a ${currentMode} document!`;
            uploadSuccess.classList.remove('hidden');
            // New document uploaded — old conversation no longer applies
            // to it, so reset memory for this mode.
            modeStates[currentMode].chatHistory = [];
        }
    } catch (error) {
        uploadStatus.classList.add('hidden');
        alert("Failed to connect to the server. Is the backend running?");
    }
});

// Configure Marked.js to use Highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
});

// Custom renderer to add copy buttons to code blocks
const renderer = new marked.Renderer();
renderer.code = function(code_or_token, language, escaped) {
    // Handle Marked v13+ (token object) vs older versions (positional args)
    let code = typeof code_or_token === 'string' ? code_or_token : code_or_token.text;
    let lang = typeof code_or_token === 'string' ? language : code_or_token.lang;
    
    let validLanguage = 'plaintext';
    let highlightedCode = escapeHTML(code);
    
    try {
        if (lang && hljs.getLanguage(lang)) {
            validLanguage = lang;
            highlightedCode = hljs.highlight(code, { language: validLanguage }).value;
        } else {
            highlightedCode = hljs.highlightAuto(code).value;
        }
    } catch (e) {
        // Fallback if highlight.js crashes
        highlightedCode = escapeHTML(code);
    }
    
    // Create a safe ID for the copy button
    const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
    
    // We store the raw code in a hidden div to easily copy it
    return `
        <div class="code-container">
            <div class="code-header">
                <span class="lang-label">${validLanguage}</span>
                <button class="copy-btn" onclick="copyCode('${codeId}')">
                    <i class="fa-regular fa-copy"></i> Copy
                </button>
            </div>
            <div id="${codeId}-raw" class="hidden">${escapeHTML(code)}</div>
            <pre><code class="hljs language-${validLanguage}">${highlightedCode}</code></pre>
        </div>
    `;
};
marked.use({ renderer });

function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// Global copy function for the inline buttons
window.copyCode = function(codeId) {
    const rawCode = document.getElementById(codeId + '-raw').innerText;
    navigator.clipboard.writeText(rawCode).then(() => {
        // Find the button and show success
        const btn = document.querySelector(`[onclick="copyCode('${codeId}')"]`);
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        setTimeout(() => { btn.innerHTML = originalText; }, 2000);
    });
};

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-msg' : 'system-msg'}`;
    
    const icon = isUser ? 'fa-user' : 'fa-robot';
    
    // Parse markdown if it's the system, else escape user input slightly
    let htmlContent = isUser ? escapeHTML(text).replace(/\n/g, '<br>') : marked.parse(text);
    
    msgDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid ${icon}"></i></div>
        <div class="msg-bubble">
            ${htmlContent}
        </div>
    `;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Add user message to UI
    addMessage(text, true);
    chatInput.value = '';

    // Show loading indicator
    const loadingId = "loading-" + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message system-msg';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-bubble">
            <div class="loader" style="width: 15px; height: 15px; border-width: 2px;"></div>
        </div>
    `;
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // BUG 3 FIX: pull this mode's running history and send it along with
    // the new message, instead of sending just the latest message alone.
    const history = modeStates[currentMode].chatHistory;

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                book_type: currentMode,
                history: history
            })
        });

        const data = await response.json();
        
        // Remove loading
        document.getElementById(loadingId).remove();

        if (data.error) {
            addMessage("❌ Error: " + data.error);
        } else {
            addMessage(data.answer);
            // BUG 3 FIX: record this exchange in the mode's history so
            // the next message includes it as context.
            history.push({ role: "user", content: text });
            history.push({ role: "assistant", content: data.answer });
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage("❌ Failed to connect to the server. Make sure FastAPI is running.");
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
