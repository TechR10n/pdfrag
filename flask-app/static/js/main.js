// Flash message handling
function showFlashMessage(message, type = 'info') {
    const flashContainer = document.querySelector('.flash-messages');
    if (!flashContainer) {
        console.warn('Flash message container not found');
        return;
    }

    const flashMessage = document.createElement('div');
    flashMessage.className = `flash-message alert alert-${type} alert-dismissible fade show`;
    flashMessage.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    flashContainer.appendChild(flashMessage);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        flashMessage.classList.remove('show');
        setTimeout(() => flashMessage.remove(), 150);
    }, 5000);
}

// Loading spinner
function showLoading(message = 'Loading...') {
    const spinner = document.createElement('div');
    spinner.className = 'spinner-overlay';
    spinner.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">${message}</p>
        </div>
    `;
    document.body.appendChild(spinner);
    return spinner;
}

function hideLoading(spinner) {
    if (spinner) {
        spinner.remove();
    }
}

// Form validation
function validateForm(form, rules) {
    const inputs = form.querySelectorAll('input, textarea, select');
    let isValid = true;

    inputs.forEach(input => {
        const fieldRules = rules[input.name];
        if (!fieldRules) return;

        // Required validation
        if (fieldRules.required && !input.value.trim()) {
            isValid = false;
            showFieldError(input, 'This field is required');
            return;
        }

        // Pattern validation
        if (fieldRules.pattern && !fieldRules.pattern.test(input.value)) {
            isValid = false;
            showFieldError(input, fieldRules.message || 'Invalid input');
            return;
        }

        // Clear any existing error
        clearFieldError(input);
    });

    return isValid;
}

function showFieldError(input, message) {
    const formGroup = input.closest('.form-group') || input.parentElement;
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    input.classList.add('is-invalid');
    formGroup.appendChild(errorDiv);
}

function clearFieldError(input) {
    const formGroup = input.closest('.form-group') || input.parentElement;
    const errorDiv = formGroup.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
    input.classList.remove('is-invalid');
}

// File upload handling
function handleFileUpload(input, previewElement, maxSize = 16 * 1024 * 1024) {
    const file = input.files[0];
    if (!file) return false;

    // Check file size
    if (file.size > maxSize) {
        showFlashMessage(`File size exceeds ${maxSize / (1024 * 1024)}MB limit`, 'danger');
        input.value = '';
        return false;
    }

    // Check file type
    if (!file.type.startsWith('application/pdf')) {
        showFlashMessage('Only PDF files are allowed', 'danger');
        input.value = '';
        return false;
    }

    // Update preview
    if (previewElement) {
        previewElement.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="bi bi-file-earmark-pdf text-danger me-2"></i>
                <span>${file.name}</span>
            </div>
        `;
    }

    return true;
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

// Export functions
window.app = {
    showFlashMessage,
    showLoading,
    hideLoading,
    validateForm,
    handleFileUpload
}; 