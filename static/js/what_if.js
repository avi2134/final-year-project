document.addEventListener("DOMContentLoaded", () => {
    const today = new Date();
    const maxFutureDays = 30;
    const futureLimit = new Date(today);
    futureLimit.setDate(today.getDate() + maxFutureDays);

    const formatDate = (d) => d.toISOString().split("T")[0];

    const investmentInput = document.getElementById("investment-date");
    const endInput = document.getElementById("end-date");

    investmentInput.setAttribute("max", formatDate(today));
    endInput.setAttribute("max", formatDate(futureLimit));

    investmentInput.addEventListener("change", () => {
        const selected = new Date(investmentInput.value);
        const minEnd = selected > today ? today : selected;
        endInput.setAttribute("min", formatDate(minEnd));
    });

    // Spinner + Fun Finance Facts
    const form = document.getElementById("analysis-form");
    const loadingSection = document.getElementById("loading-section");
    const factElement = document.getElementById("finance-fact");

    const facts = [
        "💸 Did you know? The NYSE was founded in 1792.",
        "📈 The S&P 500 was created in 1957.",
        "🧠 Warren Buffett bought his first stock at age 11.",
        "🚀 Apple became the first $3 trillion company.",
        "📊 90% of stock market gains occur in just 10% of trading days.",
        "💰 In 1986, Berkshire Hathaway shares were $2,500. Today? Over $500,000."
    ];

    let factInterval;
    form.addEventListener("submit", () => {
        loadingSection.style.display = "block";
        let index = 0;
        factElement.textContent = facts[index];
        factInterval = setInterval(() => {
            index = (index + 1) % facts.length;
            factElement.textContent = facts[index];
        }, 2500);
    });
});
