document.addEventListener("DOMContentLoaded", () => {
    const content = document.getElementById("content");

    const pages = {
        predict: `
            <h2>Make a Prediction</h2>
            <form id="predict-form">
                <label>Select model:</label>
                <select id="model-select"></select><br><br>

                <label>Upload image:</label>
                <input type="file" id="image-input" accept="image/*"><br><br>

                <div id="preview-wrap" style="display:none">
                    <strong>Preview:</strong><br>
                    <img id="image-preview" alt="preview" style="max-width:300px; max-height:300px; display:block; margin:10px 0;">
                </div>

                <button type="submit" id="predict-btn">Predict</button>
                <span id="predict-loading" style="display:none; margin-left:10px;">⏳ Predicting...</span>
            </form>

            <div id="prediction-result" style="margin-top:20px;"></div>
        `,
        models: `
            <h2>Model Information</h2>
            <div id="model-info">Loading...</div>
        `,
        dataset: `
            <h2>Dataset Information</h2>
            <p>Classes and sample images will be shown here.</p>
        `
    };

    function loadPage(name) {
        content.innerHTML = pages[name] || "<p>Page not found.</p>";
        if (name === "predict") {
            loadModels();
            attachPredictFormHandler();
        }
        if (name === "models") loadModelInfo();
    }

    function loadModels() {
        fetch("/api/models")
            .then(res => res.json())
            .then(data => {
                const select = document.getElementById("model-select");
                if (!select) return;
                select.innerHTML = ""; // clear
                if (data.available_models && data.available_models.length) {
                    data.available_models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m;
                        opt.textContent = m;
                        select.appendChild(opt);
                    });
                } else {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No models available";
                    select.appendChild(opt);
                }
            })
            .catch(err => {
                console.error("Failed to load models:", err);
                const select = document.getElementById("model-select");
                if (select) {
                    select.innerHTML = `<option value="">Error loading models</option>`;
                }
            });
    }

    function loadModelInfo() {
        fetch("/api/model_info")
            .then(res => res.json())
            .then(data => {
                const el = document.getElementById("model-info");
                el.textContent = "";
                const pre = document.createElement("pre");
                pre.textContent = JSON.stringify(data, null, 2);
                el.appendChild(pre);
            })
            .catch(err => {
                document.getElementById("model-info").textContent = "Failed to load model info.";
                console.error(err);
            });
    }

    // Converts an input File to a dataURL (base64)
    function fileToDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = (e) => reject(e);
            reader.readAsDataURL(file);
        });
    }

    // Attaches handlers for the predict form (preview + submit)
    function attachPredictFormHandler() {
        const form = document.getElementById("predict-form");
        if (!form) return;

        const fileInput = document.getElementById("image-input");
        const previewWrap = document.getElementById("preview-wrap");
        const previewImg = document.getElementById("image-preview");
        const predictBtn = document.getElementById("predict-btn");
        const loadingSpan = document.getElementById("predict-loading");
        const resultDiv = document.getElementById("prediction-result");

        // Preview when file selected
        fileInput.addEventListener("change", async () => {
            const f = fileInput.files[0];
            if (!f) {
                previewWrap.style.display = "none";
                previewImg.src = "";
                return;
            }
            try {
                const dataUrl = await fileToDataURL(f);
                previewImg.src = dataUrl;
                previewWrap.style.display = "block";
            } catch (err) {
                console.error("Failed to read file:", err);
                previewWrap.style.display = "none";
            }
        });

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            resultDiv.innerHTML = "";
            const modelSelect = document.getElementById("model-select");
            const modelName = modelSelect ? modelSelect.value : null;
            const file = fileInput.files[0];

            // Basic validation
            if (!modelName) {
                resultDiv.textContent = "Please select a model.";
                return;
            }
            if (!file) {
                resultDiv.textContent = "Please upload an image.";
                return;
            }

            // Prepare UI for loading
            predictBtn.disabled = true;
            loadingSpan.style.display = "inline";

            try {
                // Convert to base64 dataURL (this includes mime-type prefix)
                const imageDataUrl = await fileToDataURL(file);

                // Build payload - the ML service expects {image: "...", model_name: "..."}
                const payload = {
                    image: imageDataUrl,
                    model_name: modelName
                };

                const resp = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                // Try to parse JSON safely
                let json;
                try {
                    json = await resp.json();
                } catch (err) {
                    const text = await resp.text();
                    throw new Error(`Non-JSON response: ${resp.status} ${text}`);
                }

                if (!resp.ok) {
                    // backend signaled error
                    throw new Error(json.error || `Server returned ${resp.status}`);
                }

                // Display result - pretty print
                renderPredictionResult(json, resultDiv, previewImg.src, modelName);
            } catch (err) {
                console.error("Prediction error:", err);
                resultDiv.innerHTML = `<div style="color:crimson">Error: ${escapeHtml(err.message || String(err))}</div>`;
            } finally {
                predictBtn.disabled = false;
                loadingSpan.style.display = "none";
            }
        });
    }

    // Helper: render prediction nicely. Tries to extract common fields (label/class & confidence)
    function renderPredictionResult(json, container, imageSrc=null, modelName=null) {
        container.innerHTML = "";

        // small header
        const header = document.createElement("div");
        header.innerHTML = `<strong>Model:</strong> ${escapeHtml(modelName || "unknown")}`;
        container.appendChild(header);

        // If JSON contains common keys, display them prominently
        // heuristics: `prediction`, `predicted_class`, `label`, `class`, `scores`, `probs`, `confidence`
        let topText = "";
        if (json.predicted_class !== undefined) {
            topText = `Predicted class: ${escapeHtml(String(json.predicted_class))}`;
            if (json.confidence !== undefined) topText += ` (confidence: ${Number(json.confidence).toFixed(3)})`;
        } else if (json.label !== undefined || json.class !== undefined) {
            const lbl = json.label ?? json.class;
            topText = `Predicted: ${escapeHtml(String(lbl))}`;
            if (json.confidence !== undefined) topText += ` (conf: ${Number(json.confidence).toFixed(3)})`;
        } else if (json.prediction) {
            // prediction could be array or dict
            try {
                if (Array.isArray(json.prediction) && json.prediction.length > 0) {
                    topText = `Prediction array (first item): ${escapeHtml(String(json.prediction[0]))}`;
                } else if (typeof json.prediction === "object") {
                    // maybe top-1: find max value
                    const entries = Object.entries(json.prediction);
                    if (entries.length) {
                        entries.sort((a,b) => b[1] - a[1]);
                        topText = `Top: ${escapeHtml(entries[0][0])} (${Number(entries[0][1]).toFixed(4)})`;
                    }
                } else {
                    topText = `Prediction: ${escapeHtml(String(json.prediction))}`;
                }
            } catch (err) {
                topText = `Prediction: ${escapeHtml(JSON.stringify(json.prediction))}`;
            }
        }

        if (topText) {
            const p = document.createElement("p");
            p.innerHTML = `<strong>${topText}</strong>`;
            container.appendChild(p);
        }

        // Show image and full JSON below
        const box = document.createElement("div");
        box.style.display = "flex";
        box.style.gap = "20px";
        box.style.alignItems = "flex-start";

        if (imageSrc) {
            const left = document.createElement("div");
            left.style.minWidth = "150px";
            left.innerHTML = `<img src="${imageSrc}" alt="sent image" style="max-width:200px; max-height:200px; border:1px solid #ddd; padding:4px;">`;
            box.appendChild(left);
        }

        const right = document.createElement("div");
        right.style.flex = "1";
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(json, null, 2);
        right.appendChild(pre);
        box.appendChild(right);

        container.appendChild(box);
    }

    // small helper to avoid XSS when injecting strings
    function escapeHtml(unsafe) {
        return unsafe
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    // Wire nav links
    document.querySelectorAll("nav a").forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const page = link.getAttribute("data-page");
            loadPage(page);
        });
    });

    // Load default page
    loadPage("predict");
});
