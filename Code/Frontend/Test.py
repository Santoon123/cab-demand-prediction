$(document).ready(function () {
    setTimeout(function () {
        $("#intro-container").css("opacity", "0");
        setTimeout(function () {
            $("#intro-container").css("display", "none");
            $("#parent").css("display", "flex");
            $("#datepicker").datepicker();

            // ✅ Initialize Cesium
            Cesium.Ion.defaultAccessToken =
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5NDM0MjhmMi1hYjU3LTQ4ODQtYmMwOS03MjhhMDUyM2JmODQiLCJpZCI6Mjc2NjgzLCJpYXQiOjE3Mzk4MDA2NDJ9.n4Yhbtdq6trE4vfIjnG5NYmbqM7AdB64Tie_NSBbGj4";

            const viewer = new Cesium.Viewer("cesiumContainer", {
                terrain: Cesium.Terrain.fromWorldTerrain(),
                geocoder: false,
            });

            viewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(
                    -74.006,
                    40.7128,
                    5000
                ),
                orientation: {
                    heading: Cesium.Math.toRadians(0.0),
                    pitch: Cesium.Math.toRadians(-90.0),
                },
            });

            async function addBuildings() {
                const buildingTileset = await Cesium.createOsmBuildingsAsync();
                viewer.scene.primitives.add(buildingTileset);
            }
            addBuildings();
            const geocoder = new Cesium.Geocoder({
                container: document.getElementById("geocoderContainer"),
                scene: viewer.scene,
                destinationFound: async function (viewModel, destination) {
                    console.log("✅ Geocode completed!");

                    let centerLon, centerLat;
                    if (destination.west !== undefined) {
                        centerLon = (destination.west + destination.east) / 2;
                        centerLat = (destination.south + destination.north) / 2;
                        centerLon = Cesium.Math.toDegrees(centerLon);
                        centerLat = Cesium.Math.toDegrees(centerLat);
                    } else {
                        let cartographic =
                            Cesium.Cartographic.fromCartesian(destination);
                        centerLat = Cesium.Math.toDegrees(
                            cartographic.latitude
                        );
                        centerLon = Cesium.Math.toDegrees(
                            cartographic.longitude
                        );
                    }

                    let locationID = await getPULocationID(
                        centerLat,
                        centerLon
                    );
                    viewer.camera.flyTo({
                        destination: Cesium.Cartesian3.fromDegrees(
                            centerLon,
                            centerLat,
                            2000
                        ),
                        orientation: {
                            heading: Cesium.Math.toRadians(0.0),
                            pitch: Cesium.Math.toRadians(-45.0), // Slightly tilted view
                        },
                        duration: 3, // Fly duration in seconds
                    });
                    geocoder.viewModel.search();
                    if (locationID) {
                        console.log(`📍 Mapped to PULocationID: ${locationID}`);
                        document.getElementById("locationpicker").value =
                            locationID;
                    }
                },
            });
            console.log("✅ Manual Geocoder added!");
            let predictButton = document.getElementById("predictButton");
            if (predictButton) {
                predictButton.addEventListener("click", function () {
                    console.log("✅ Predict Demand button clicked!");
                    fetchDemandPrediction();
                });
            } else {
                console.error("❌ Predict Button not found in the DOM.");
            }
        }, 500);
    }, 4000);
});
async function getPULocationID(lat, lon) {
    try {
        let response = await fetch(
            `http://127.0.0.1:5000/get_zone?lat=${lat}&lon=${lon}`
        );
        let data = await response.json();
        console.log("✅ API Response:", data);
        return data.locationID || null;
    } catch (error) {
        console.error("❌ Error fetching PULocationID:", error);
        return null;
    }
}
async function fetchDemandPrediction() {
    let date = document.getElementById("datepicker").value;
    let time = document.getElementById("timepicker").value;
    let locationID = document.getElementById("locationpicker").value;

    console.log(
        "📅 Date:",
        date,
        "⏰ Time:",
        time,
        "📍 Location ID:",
        locationID
    );

    if (!date || !time || !locationID) {
        alert("❌ Please enter all fields!");
        return;
    }
    let queryUrl = `http://127.0.0.1:5000/predict_demand?date=${date}&time=${time}&locationID=${locationID}`;
    try {
        let response = await fetch(queryUrl);
        let data = await response.json();
        console.log("✅ Prediction Response:", data);

        document.getElementById("predictionResult").innerHTML = `
            🚖 Predicted Demand: <b>${data.num_rides}</b> rides <br>
            💰 Total Fare Collected: <b>$${data.total_fare}</b><br>
            🤑 Estimated Avg Income: <b>$${data.avg_income}</b>
        `;
    } catch (error) {
        console.error("❌ Error fetching demand prediction:", error);
        document.getElementById("predictionResult").innerHTML =
            "⚠️ Error fetching demand prediction.";
    }
}
