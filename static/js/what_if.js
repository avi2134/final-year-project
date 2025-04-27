document.addEventListener("DOMContentLoaded", () => {
    const today = new Date();
    const maxFutureDays = 30;
    const futureLimit = new Date(today);
    futureLimit.setDate(today.getDate() + maxFutureDays);

    const formatDate = (d) => d.toISOString().split("T")[0];

    const investmentInput = document.getElementById("investment-date");
    const endInput = document.getElementById("end-date");
    const form = document.getElementById("analysis-form");
    const loadingSection = document.getElementById("loading-section");

    if (investmentInput && endInput) {
        investmentInput.setAttribute("max", formatDate(today));
        endInput.setAttribute("max", formatDate(futureLimit));

        investmentInput.addEventListener("change", () => {
            const selected = new Date(investmentInput.value);
            const minEnd = selected > today ? today : selected;
            endInput.setAttribute("min", formatDate(minEnd));
        });
    }

    // Show loading if form is submitted
    form?.addEventListener("submit", () => {
        const endDateValue = new Date(document.getElementById("end-date").value);
        if (endDateValue > today) {
            loadingSection.style.display = "block";
        }
    });

    // If redirected with task_id, show loading and start polling
    const params = new URLSearchParams(window.location.search);
    const taskId = params.get("task_id");

    if (taskId) {
        if (!localStorage.getItem("pendingTaskId")) {
            localStorage.setItem("pendingTaskId", taskId);
            console.log("Saving task id:", taskId);
        }
        loadingSection.style.display = "block";

        // Polling for task status
        console.log("Polling for task:", taskId);

        function checkTaskStatus() {
            fetch(`/what-if-status/?task_id=${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === "completed" && data.result_url) {
                        console.log("Task completed! Redirecting...");
                        showRedirectingMessage();
                        setTimeout(() => {
                            window.location.href = data.result_url;
                        }, 3000); // wait 3 seconds before redirect
                        clearInterval(pollingInterval);
                    } else if (data.status === "failed") {
                        alert("Sorry, your analysis failed. Please try again later.");
                        clearInterval(pollingInterval);
                    }
                })
                .catch(error => {
                    console.error("Error polling task:", error);
                });
        }

        function showRedirectingMessage() {
            const statusElement = document.querySelector(".status");
            if (statusElement) {
                statusElement.innerText = "Analysis complete! Redirecting you to your results...";
            }
        }

        const pollingInterval = setInterval(checkTaskStatus, 5000); // Poll every 5 seconds
    }
});