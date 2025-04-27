document.addEventListener("DOMContentLoaded", () => {
    const pending = "{{ pending|yesno:'true,false' }}" === "true";  // from Django template
    const taskId = "{{ task_id|default:'' }}";

    if (pending && taskId) {
        // Start polling every 5 seconds
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/what-if-status/?task_id=${taskId}`);
                const data = await response.json();

                if (data.ready) {
                    clearInterval(interval);
                    location.reload();  // Refresh page to show results
                } else if (data.error) {
                    clearInterval(interval);
                    document.getElementById("loading-section").innerHTML = `
                        <div class="alert alert-danger mt-4">${data.error}</div>
                    `;
                }
            } catch (err) {
                console.error("Error checking task status:", err);
            }
        }, 5000);  // 5 seconds
    }
});