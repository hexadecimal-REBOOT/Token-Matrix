const API_BASE = (() => {
  if (window.__VDA_API_BASE__) return window.__VDA_API_BASE__;
  const protocol = window.location.protocol;
  const host = window.location.hostname;
  const defaultPort = protocol === 'https:' ? 443 : 8000;
  const port = window.__VDA_API_PORT__ || defaultPort;
  return `${protocol}//${host}:${port}`;
})();

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const clearButton = document.getElementById('clearFiles');
const fileList = document.getElementById('fileList');
const fileSummary = document.getElementById('fileSummary');
const statusEl = document.getElementById('status');
const previewEl = document.getElementById('preview');
const previewContentEl = document.getElementById('previewContent');
const configForm = document.getElementById('configForm');
const convertButton = document.getElementById('convertButton');

const fileTemplate = document.getElementById('fileRowTemplate');

/** @type {Map<string, File>} */
const storedFiles = new Map();

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function updateSummary() {
  if (storedFiles.size === 0) {
    fileSummary.hidden = true;
    fileSummary.textContent = '';
    return;
  }
  const totalBytes = [...storedFiles.values()].reduce((acc, file) => acc + file.size, 0);
  fileSummary.hidden = false;
  fileSummary.textContent = `${storedFiles.size} file${storedFiles.size === 1 ? '' : 's'} • ${formatBytes(totalBytes)}`;
}

function renderFileList() {
  fileList.innerHTML = '';
  const fragment = document.createDocumentFragment();

  [...storedFiles.entries()]
    .sort(([pathA], [pathB]) => pathA.localeCompare(pathB))
    .forEach(([path, file]) => {
      const node = fileTemplate.content.firstElementChild.cloneNode(true);
      node.querySelector('.file-name').textContent = path;
      node.querySelector('.file-size').textContent = formatBytes(file.size);
      node.dataset.path = path;
      fragment.appendChild(node);
    });

  fileList.appendChild(fragment);
  updateSummary();
}

function showStatus(message, kind = 'info') {
  statusEl.textContent = message;
  statusEl.classList.remove('success', 'error');
  if (kind === 'success') statusEl.classList.add('success');
  if (kind === 'error') statusEl.classList.add('error');
}

function clearStatus() {
  statusEl.textContent = '';
  statusEl.classList.remove('success', 'error');
}

function resetPreview() {
  previewEl.hidden = true;
  previewContentEl.textContent = '';
}

function handleFiles(newFiles) {
  let added = 0;
  for (const { file, path } of newFiles) {
    if (!storedFiles.has(path)) {
      storedFiles.set(path, file);
      added++;
    } else {
      storedFiles.set(path, file); // replace updated file
    }
  }
  if (added > 0) {
    showStatus(`Added ${added} file${added === 1 ? '' : 's'} to the queue.`);
  }
  renderFileList();
}

function extractPath(file) {
  return file.webkitRelativePath || file.relativePath || file.name;
}

async function traverseEntry(entry, parentPath = '') {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      entry.file((file) => {
        const path = parentPath ? `${parentPath}/${file.name}` : file.name;
        resolve([{ file, path }]);
      }, reject);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const entries = [];
      const readEntries = () => {
        reader.readEntries(async (batch) => {
          if (!batch.length) {
            const nestedPromises = entries.map((child) => traverseEntry(child, parentPath ? `${parentPath}/${entry.name}` : entry.name));
            const nestedResults = await Promise.all(nestedPromises);
            resolve(nestedResults.flat());
            return;
          }
          entries.push(...batch);
          readEntries();
        }, reject);
      };
      readEntries();
    } else {
      resolve([]);
    }
  });
}

async function getFilesFromDataTransfer(dataTransfer) {
  const items = [...dataTransfer.items];
  const collected = [];
  const directoryPromises = [];

  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      directoryPromises.push(traverseEntry(entry));
    } else {
      const file = item.getAsFile();
      if (file) {
        collected.push({ file, path: extractPath(file) });
      }
    }
  }

  const directories = await Promise.all(directoryPromises);
  directories.flat().forEach((fileObj) => collected.push(fileObj));
  return collected;
}

async function previewFile(path) {
  const file = storedFiles.get(path);
  if (!file) return;

  if (file.size > 1024 * 256) {
    previewContentEl.textContent = 'File too large to preview. Download and inspect locally.';
    previewEl.hidden = false;
    return;
  }

  const textTypes = ['application/json', 'text/plain', 'application/x-yaml', 'application/octet-stream'];
  const isText = textTypes.some((type) => file.type.startsWith(type)) || /\.(json|txt|yaml|yml|py|md)$/i.test(path);

  if (!isText) {
    previewContentEl.textContent = 'Binary file preview is not supported.';
    previewEl.hidden = false;
    return;
  }

  const text = await file.text();
  previewContentEl.textContent = text || '[Empty file]';
  previewEl.hidden = false;
}

function removeFile(path) {
  storedFiles.delete(path);
  renderFileList();
  if (storedFiles.size === 0) {
    clearStatus();
    resetPreview();
  }
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

dropzone.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragover');
  const files = await getFilesFromDataTransfer(event.dataTransfer);
  handleFiles(files);
});

fileInput.addEventListener('change', () => {
  const files = [...fileInput.files].map((file) => ({ file, path: extractPath(file) }));
  handleFiles(files);
  fileInput.value = '';
});

fileList.addEventListener('click', (event) => {
  const item = event.target.closest('.file-item');
  if (!item) return;
  const { path } = item.dataset;
  if (!path) return;

  if (event.target.matches('.preview-button')) {
    previewFile(path);
  }

  if (event.target.matches('.remove-button')) {
    removeFile(path);
  }
});

clearButton.addEventListener('click', () => {
  storedFiles.clear();
  renderFileList();
  clearStatus();
  resetPreview();
});

configForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (storedFiles.size === 0) {
    showStatus('Add at least one checkpoint file before converting.', 'error');
    return;
  }

  const formData = new FormData(configForm);
  for (const [path, file] of storedFiles.entries()) {
    formData.append('files', file, path);
  }

  convertButton.disabled = true;
  showStatus('Uploading checkpoint and starting conversion…');
  resetPreview();

  try {
    const response = await fetch(`${API_BASE.replace(/\/$/, '')}/api/convert`, {
      method: 'POST',
      body: formData,
    });

    const contentType = response.headers.get('content-type') || '';
    if (!response.ok) {
      let errorMessage = `Conversion failed with status ${response.status}.`;
      if (contentType.includes('application/json')) {
        const data = await response.json();
        if (data?.detail) {
          errorMessage = data.detail;
        }
      } else {
        const text = await response.text();
        if (text) errorMessage = text;
      }
      throw new Error(errorMessage);
    }

    if (contentType.includes('application/json')) {
      const payload = await response.json();
      if (payload?.detail) {
        throw new Error(payload.detail);
      }
      throw new Error('Unexpected JSON response from server.');
    }

    const blob = await response.blob();
    const filename = response.headers.get('x-export-filename') || 'video_depth_anything.onnx';
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    showStatus('Conversion succeeded! Download should begin automatically.', 'success');
  } catch (error) {
    console.error(error);
    showStatus(error.message || 'Conversion failed due to an unknown error.', 'error');
  } finally {
    convertButton.disabled = false;
  }
});

renderFileList();
clearStatus();
resetPreview();
