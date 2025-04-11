document.addEventListener("DOMContentLoaded", function () {
    function fetchLeaderboard() {
        fetch('/api/get-leaderboard/')
            .then(response => response.json())
            .then(data => {
                let leaderboardContainer = document.getElementById("leaderboard");
                leaderboardContainer.innerHTML = ""; // Clear previous entries

                if (data.leaderboard.length === 0) {
                    leaderboardContainer.innerHTML = "<p class='text-center'>No leaderboard data available.</p>";
                    return;
                }

                let leaderboardHTML = `
                    <div class="container">
                        <div class="row">
                            <div class="col-md-8 offset-md-2">
                                <table class="table table-bordered table-hover text-center shadow-sm">
                                    <thead class="table-dark">
                                        <tr>
                                            <th class="p-3">Rank</th>
                                            <th class="p-3">User</th>
                                            <th class="p-3">XP</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                `;

                data.leaderboard.forEach(player => {
                    leaderboardHTML += `
                        <tr>
                            <td class="p-3 fw-bold">#${player.rank}</td>
                            <td class="p-3">${player.username}</td>
                            <td class="p-3">${player.xp}</td>
                        </tr>
                    `;
                });

                leaderboardHTML += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                `;

                leaderboardContainer.innerHTML = leaderboardHTML;
            })
            .catch(error => console.error("Error fetching leaderboard:", error));
    }

    fetchLeaderboard();
});