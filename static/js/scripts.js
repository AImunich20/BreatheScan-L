// 🔹 สลับหน้า
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');

    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    document.querySelector(`[onclick="showPage('${pageId}')"]`).classList.add('active');
}

// 🔹 Quiz
function checkQuiz() {
    let answer = document.querySelector('input[name="q1"]:checked');
    let result = document.getElementById("quiz-result");
    if (!answer) {
        result.innerHTML = `<p class="text-red-600">Please select an answer!</p>`;
    } else if (answer.value === "b") {
        result.innerHTML = `<p class="text-green-600 font-bold">Correct! Obesity and diabetes are risk factors.</p>`;
    } else {
        result.innerHTML = `<p class="text-red-600">Incorrect. Try again!</p>`;
    }
    result.classList.remove("hidden");
}

// 🔹 Progress Bar + Step control
function nextStep(step) {
    document.querySelectorAll(".test-step").forEach(s => s.classList.add("hidden"));
    document.getElementById(`step${step}`).classList.remove("hidden");

    let progress = document.getElementById("progress");
    progress.style.width = `${(step - 1) * 33}%`;
}

// 🔹 ใช้ simulated data แทน CSV
function useSimulatedData() {
    alert("Simulated breath data uploaded successfully!");
}

// 🔹 Chart.js ROC Curve
document.addEventListener("DOMContentLoaded", () => {
    let ctx = document.getElementById("rocChart");
    if (ctx) {
        new Chart(ctx, {
            type: "line",
            data: {
                labels: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
                datasets: [{
                    label: "ROC Curve",
                    data: [0, 0.6, 0.8, 0.9, 0.95, 0.97, 1],
                    borderColor: "#4f46e5",
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: "False Positive Rate" } },
                    y: { title: { display: true, text: "True Positive Rate" } }
                }
            }
        });
    }
});
document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("mobile-menu-button");
    const menu = document.getElementById("mobile-menu");

    if (btn && menu) {
        btn.addEventListener("click", () => {
            menu.classList.toggle("hidden");
        });
    }
});
