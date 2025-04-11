document.addEventListener("DOMContentLoaded", function () {
    // Fetch Latest Financial News
    fetchNews(); // Load default business news on page load

    document.getElementById("search-news-btn").addEventListener("click", function () {
        let query = document.getElementById("news-search").value.trim();
        fetchNews(query); // Fetch news dynamically based on user search
    });

    fetchTrendingStocks();
});

function fetchTrendingStocks() {
    fetch('/fetch-trending-stocks/')
    .then(response => response.json())
    .then(data => {
        console.log("Trending Stocks API Response:", data);

        let stockList = document.getElementById("trending-stocks");
        stockList.innerHTML = "";

        if (!data.trending || !Array.isArray(data.trending)) {
            console.error("Invalid trending stocks data:", data);
            stockList.innerHTML = "<li class='list-group-item text-warning'>No trending stocks available.</li>";
            return;
        }

        data.trending.forEach(stock => {
            let li = document.createElement("li");
            li.classList.add("list-group-item");
            li.innerHTML = `<strong>${stock.symbol}</strong>: ${stock.price !== "N/A" ? `$${stock.price.toFixed(2)}` : "Price Unavailable"}`;
            stockList.appendChild(li);
        });
    })
    .catch(error => console.error("Error fetching trending stocks:", error));
}

function fetchNews(query = "") {
    fetch(`/fetch-news/?query=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            let newsList = document.getElementById("news-list");
            newsList.innerHTML = "";

            if (!data.articles || data.articles.length === 0) {
                newsList.innerHTML = "<li class='list-group-item text-warning'>No news found.</li>";
                return;
            }

            data.articles.forEach(article => {
                let li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `<a href="${article.url}" target="_blank">${article.title}</a>`;
                newsList.appendChild(li);
            });
        })
        .catch(error => console.error("Error fetching news:", error));
}