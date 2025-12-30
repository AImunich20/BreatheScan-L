function createPercentChart(ctx, label, color) {
    return new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label,
                data: [],
                borderColor: color,
                tension: 0.35,
                fill: false
            }]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: value => value + "%"
                    }
                }
            }
        }
    });
}

const cpuChart = createPercentChart(
    document.getElementById("cpuChart"),
    "CPU %",
    "#2563eb"
);

const ramChart = createPercentChart(
    document.getElementById("ramChart"),
    "RAM %",
    "#16a34a"
);

const diskChart = createPercentChart(
    document.getElementById("diskChart"),
    "Disk %",
    "#ea580c"
);

const netChart = new Chart(document.getElementById("netChart"), {
    type: "line",
    data: {
        labels: [],
        datasets: [
            {
                label: "Sent MB",
                data: [],
                borderColor: "#9333ea",
                tension: 0.35
            },
            {
                label: "Recv MB",
                data: [],
                borderColor: "#4f46e5",
                tension: 0.35
            }
        ]
    },
    options: {
        responsive: true,
        animation: false
    }
});

function pushData(chart, values) {
    chart.data.labels.push("");
    chart.data.datasets.forEach((ds, i) => {
        ds.data.push(Array.isArray(values) ? values[i] : values);
        if (ds.data.length > 25) ds.data.shift();
    });
    if (chart.data.labels.length > 25) chart.data.labels.shift();
    chart.update();
}

setInterval(() => {
    fetch("/api/admin/server_status")
        .then(r => r.json())
        .then(d => {

            // Text summary
            document.getElementById("cpuText").innerText = d.cpu + "%";
            document.getElementById("ramText").innerText = d.ram.percent + "%";
            document.getElementById("diskText").innerText = d.disk.percent + "%";
            document.getElementById("netText").innerText =
                `${d.net.sent} / ${d.net.recv}`;

            // Charts
            pushData(cpuChart, d.cpu);
            pushData(ramChart, d.ram.percent);
            pushData(diskChart, d.disk.percent);
            pushData(netChart, [d.net.sent, d.net.recv]);
        });
}, 2000);
