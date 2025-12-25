function toggleUserDropdown() {
    const menu = document.getElementById("userDropdown");
    if (!menu) return;

    menu.classList.toggle("hidden");
}

// ปิด dropdown เมื่อคลิกข้างนอก
document.addEventListener("click", function (e) {
    const menu = document.getElementById("userDropdown");
    const userMenu = document.getElementById("userMenu");
    const button = userMenu?.querySelector("button");

    if (!menu || !button) return;

    if (!menu.contains(e.target) && !button.contains(e.target)) {
        menu.classList.add("hidden");
    }
});

// ---------- LOGIN ----------
function openLoginModal() {
    const modal = document.getElementById("loginModal");
    modal.classList.remove("hidden");
    modal.classList.add("flex");
}

function closeLoginModal() {
    const modal = document.getElementById("loginModal");
    modal.classList.add("hidden");
    modal.classList.remove("flex");
}

// ---------- REGISTER ----------
function openRegisterModal() {
    closeLoginModal();
    const modal = document.getElementById("registerModal");
    modal.classList.remove("hidden");
    modal.classList.add("flex");
}

function closeRegisterModal() {
    const modal = document.getElementById("registerModal");
    modal.classList.add("hidden");
    modal.classList.remove("flex");
}

// ---------- LOGIN SUBMIT ----------
async function loginUser(e) {
    e.preventDefault();

    const username = document.getElementById("login_username").value;
    const password = document.getElementById("login_password").value;

    const res = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
    });

    const data = await res.json();

    if (data.success) {
        location.reload();
    } else {
        alert(data.msg || "Login failed");
    }
}
function logoutUser() {
    window.location.href = "/logout";
}

function toggleUserDropdown() {
    document.getElementById("userDropdown").classList.toggle("hidden");
}

function openAccountPopup(type) {
    const popup = document.getElementById("accountPopup");
    const content = document.getElementById("accountPopupContent");

    popup.classList.remove("hidden");

    if (type === "profile") {
        content.innerHTML = `
            <p><b>ชื่อ:</b> {{ user.first_name }} {{ user.last_name }}</p>
            <p><b>Username:</b> {{ user.username }}</p>
            <p><b>Email:</b> {{ user.email }}</p>
            <p><b>Role:</b> {{ 'Admin' if user.adminst else 'User' }}</p>
            <p><b>สมัครเมื่อ:</b> {{ user.created_at }}</p>
        `;
    }

    if (type === "history") {
        content.innerHTML = `
            <p class="font-semibold">📜 ประวัติการใช้งาน</p>
            <ul class="list-disc pl-5">
                <li>Login ล่าสุด</li>
                <li>วิเคราะห์ข้อมูล</li>
                <li>ดาวน์โหลดรายงาน</li>
            </ul>
        `;
    }

    if (type === "settings") {
        content.innerHTML = `
            <button class="w-full bg-gray-100 hover:bg-gray-200 py-2 rounded">
                🔑 เปลี่ยนรหัสผ่าน
            </button>
            <button class="w-full bg-gray-100 hover:bg-gray-200 py-2 rounded mt-2">
                ✏️ แก้ไขโปรไฟล์
            </button>
        `;
    }
}

document.addEventListener("click", function (e) {
    const menu = document.getElementById("userMenu");
    if (!menu.contains(e.target)) {
        document.getElementById("userDropdown").classList.add("hidden");
        document.getElementById("accountPopup").classList.add("hidden");
    }
});
