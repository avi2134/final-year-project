document.addEventListener("DOMContentLoaded", function () {
    // Fetch Latest Financial News
    fetch('/fetch-news/')
        .then(response => response.json())
        .then(data => {
            let newsList = document.getElementById("news-list");
            newsList.innerHTML = "";
            data.articles.forEach(article => {
                let li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `<a href="${article.url}" target="_blank">${article.title}</a>`;
                newsList.appendChild(li);
            });
        })
        .catch(error => console.error("Error fetching news:", error));

    // Fetch Trending Stocks Dynamically
    fetch('/fetch-trending-stocks/')
        .then(response => response.json())
        .then(data => {
            let stockList = document.getElementById("trending-stocks");
            stockList.innerHTML = "";
            data.trending.forEach(stock => {
                let li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `<strong>${stock.symbol}</strong>: $${stock.price}`;
                stockList.appendChild(li);
            });
        })
        .catch(error => console.error("Error fetching trending stocks:", error));

});
