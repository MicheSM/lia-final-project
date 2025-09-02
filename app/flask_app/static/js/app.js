document.addEventListener("DOMContentLoaded", () => {
    const content = document.getElementById("content");

    const pages = {
        predict: `
            <h2>Make a Prediction</h2>
            <form id="predict-form">
                <label>Select model:</label>
                <select id="model-select"></select><br><br>
                <label>Upload image:</label>
                <input type="file" id="image-input"><br><br>
                <button type="submit">Predict</button>
            </form>
            <div id="prediction-result"></div>
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
        if (name === "predict") loadModels();
        if (name === "models") loadModelInfo();
    }

    function loadModels() {
        fetch("/api/models")
            .then(res => res.json())
            .then(data => {
                const select = document.getElementById("model-select");
                if (data.available_models) {
                    data.available_models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m;
                        opt.textContent = m;
                        select.appendChild(opt);
                    });
                }
            });
    }

    function loadModelInfo() {
        fetch("/api/model_info")
            .then(res => res.json())
            .then(data => {
                document.getElementById("model-info").textContent = JSON.stringify(data);
            });
    }

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
