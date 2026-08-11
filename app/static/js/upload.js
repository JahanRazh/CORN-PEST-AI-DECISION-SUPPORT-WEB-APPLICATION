/* Drag-and-drop uploader with client-side validation and a progress state.
   The form still posts normally, so the page works if JavaScript is disabled;
   this script only improves the experience. */

(function () {
  'use strict';

  const dropzone     = document.getElementById('dropzone');
  const input        = document.getElementById('imageInput');
  const prompt       = document.getElementById('dropPrompt');
  const preview      = document.getElementById('dropPreview');
  const previewImage = document.getElementById('previewImage');
  const previewName  = document.getElementById('previewName');
  const previewSize  = document.getElementById('previewSize');
  const clearButton  = document.getElementById('clearImage');
  const fileError    = document.getElementById('fileError');
  const form         = document.getElementById('detectForm');
  const submitButton = document.getElementById('submitButton');
  const submitLabel  = document.getElementById('submitLabel');
  const submitIcon   = document.getElementById('submitIcon');
  const scanOverlay  = document.getElementById('scanOverlay');

  if (!dropzone || !input) return;

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

  function showError(message) {
    fileError.textContent = message;
    fileError.classList.remove('hidden');
  }

  function clearError() {
    fileError.textContent = '';
    fileError.classList.add('hidden');
  }

  function humanSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function accept(file) {
    clearError();

    if (ALLOWED.indexOf(file.type) === -1) {
      showError('That file type is not supported. Use JPG, PNG, WEBP or BMP.');
      return false;
    }
    if (file.size > MAX_BYTES) {
      showError('That image is ' + humanSize(file.size) + '. The limit is 10 MB.');
      return false;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
      previewImage.src = event.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = humanSize(file.size);
      prompt.classList.add('hidden');
      preview.classList.remove('hidden');
      dropzone.classList.remove('py-12');
      dropzone.classList.add('py-6');
    };
    reader.readAsDataURL(file);
    return true;
  }

  function reset() {
    input.value = '';
    previewImage.src = '';
    preview.classList.add('hidden');
    prompt.classList.remove('hidden');
    dropzone.classList.add('py-12');
    dropzone.classList.remove('py-6');
    clearError();
  }

  /* --- Click and keyboard ------------------------------------------------ */
  dropzone.addEventListener('click', function (event) {
    if (event.target.closest('#clearImage')) return;
    input.click();
  });

  dropzone.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });

  clearButton.addEventListener('click', function (event) {
    event.stopPropagation();
    reset();
  });

  input.addEventListener('change', function () {
    if (input.files && input.files[0]) {
      if (!accept(input.files[0])) reset();
    }
  });

  /* --- Drag and drop ----------------------------------------------------- */
  ['dragenter', 'dragover'].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add('is-dragging');
    });
  });

  ['dragleave', 'drop'].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      if (name === 'dragleave' && dropzone.contains(event.relatedTarget)) return;
      dropzone.classList.remove('is-dragging');
    });
  });

  dropzone.addEventListener('drop', function (event) {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (!files || !files.length) return;
    // DataTransfer lets us put the dropped file straight onto the input, so the
    // normal form post carries it without any extra request.
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    input.files = transfer.files;
    if (!accept(files[0])) reset();
  });

  /* --- Paste from clipboard ---------------------------------------------- */
  document.addEventListener('paste', function (event) {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image/') === 0) {
        const file = items[i].getAsFile();
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        accept(file);
        dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
        break;
      }
    }
  });

  /* --- Submit state ------------------------------------------------------ */
  form.addEventListener('submit', function (event) {
    if (!input.files || !input.files.length) {
      event.preventDefault();
      showError('Choose an image before analysing.');
      dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    submitButton.disabled = true;
    submitLabel.textContent = 'Running the pipeline…';
    submitIcon.outerHTML =
      '<svg class="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">' +
      '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>' +
      '<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
    if (scanOverlay) scanOverlay.classList.remove('hidden');
  });
})();
