import requests

from planner_agent.tools.config import TRANSPORT_ADAPTERAPI_ENDPOINT, X_API_Key, Transport_Agent_Folder, S3_BUCKET
from planner_agent.tools.final_agent_helper import create_pdf_bytes_plain_from_html


def call_transport_agent_api(bucket_name: str, key: str, sender_agent: str, session: str):
    """
    Makes an API call to the specified endpoint using the provided data.
    :param bucket_name: Name of the S3 bucket
    :param key: Path to the file in the S3 bucket
    :param sender_agent: Sender agent name
    :param session: Session identifier
    :return: Response from the API as a dictionary
    """
    url = TRANSPORT_ADAPTERAPI_ENDPOINT + "/transport"
    headers = {"Content-Type": "application/json", "X-API-Key": X_API_Key}
    payload = {
        "bucket_name": bucket_name,
        "key": Transport_Agent_Folder +"/"+ key,
        "sender_agent": sender_agent,
        "session": session
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response
    except requests.RequestException as e:
        return {}


if __name__ == "__main__":
    human_text= f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body  font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; 
        h1  color: #333; 
        h2  color: #0056b3; 
        .day  background: #fff; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); 
        .place  margin-bottom: 15px; 
        .transport  background: #e7f3fe; padding: 10px; border-left: 5px solid #2196F3; margin-top: 10px; 
        .summary  background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); 
        .action-items  background: #fff; border-radius: 8px; padding: 15px; margin-top: 20px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); 
        ul  list-style: none; padding: 0; 
        li  margin: 5px 0; 
    </style>
    <title>Family Trip to Singapore (2025-06-01 to 2025-06-03)</title>
</head>
<body>
    <h1>Family Trip to Singapore (2025-06-01 to 2025-06-03)</h1>
    <p>Welcome to your exciting 3-day adventure in Singapore! Enjoy a balanced exploration of iconic attractions and unique experiences with your family.</p>

    <div class="day">
        <h2>Day 1: 2025-06-01</h2>
        <h3>Morning:</h3>
        <div class="place" data-place-id="ChIJMxZ-kwQZ2jERdsqftXeWCWI">
            <strong>Gardens by the Bay</strong>
            <p>A magnificent garden featuring futuristic structures and lush plant life, perfect for family exploration.</p>
            <p>Address: 18 Marina Gardens Dr, Singapore 018953</p>
            <h4>Why this place pick:</h4>
            <p>Highly rated for its stunning views and accessibility, it's a family-friendly choice!</p>
            <div class="transport" data-from="accommodation" data-to="ChIJMxZ-kwQZ2jERdsqftXeWCWI">
                <strong>Transport Options:</strong>
                <div>Ride (32 mins) - SGD 19.85 - Carbon: 5.05 kg - Fastest route via Grab/Taxi.</div>
                <div>MRT (61 mins) - SGD 3.66 - Carbon: 0.41 kg - Take East West Line, then Thomson East Coast Line.</div>
            </div>
        </div>
        <div class="place" data-place-id="ChIJ29omWQgZ2jEROEz2yZFzQp8">
            <strong>Merlion</strong>
            <p>The iconic symbol of Singapore, representing the city’s rich heritage and culture.</p>
            <p>Address: Singapore</p>
            <h4>Why this place pick:</h4>
            <p>This landmark is highly recommended for its photo opportunities and accessibility!</p>
            <div class="transport" data-from="ChIJMxZ-kwQZ2jERdsqftXeWCWI" data-to="ChIJ29omWQgZ2jEROEz2yZFzQp8">
                <strong>Transport Options:</strong>
                <div>Ride (12 mins) - SGD 7.56 - Carbon: 0.97 kg - Fastest route via Grab/Taxi.</div>
                <div>Cycle (10 mins) - Free - Carbon: 0 kg - Eco-friendly cycling route.</div>
            </div>
        </div>

        <h3>Lunch:</h3>
        <div class="place" data-place-id="ChIJ5Y6l4Q0Z2jERYL0KDIjT6v0">
            <strong>Lau Pa Sat</strong>
            <p>A historic food market offering a wide selection of local Singaporean dishes and international cuisine.</p>
            <p>Address: 18 Raffles Quay, Singapore 048582</p>
            <h4>Why this place pick:</h4>
            <p>Centrally located and highly rated for its diverse food options, perfect for refueling!</p>
            <div class="transport" data-from="ChIJ29omWQgZ2jEROEz2yZFzQp8" data-to="ChIJ5Y6l4Q0Z2jERYL0KDIjT6v0">
                <strong>Transport Options:</strong>
                <div>Ride (6 mins) - SGD 5.57 - Carbon: 0.41 kg - Quick shake via Grab/Taxi.</div>
                <div>Bus (18 mins) - SGD 1.18 - Carbon: 0.161 kg - Take Bus 145 for direct public transport.</div>
            </div>
        </div>

        <h3>Afternoon:</h3>
        <div class="place" data-place-id="ChIJBTYg1g4Z2jERp_MBbu5erWY">
            <strong>Merlion Park</strong>
            <p>Home to the Merlion statue, this park offers a fantastic view of Marina Bay Sands.</p>
            <p>Address: 1 Fullerton Rd, Singapore 049213</p>
            <h4>Why this place pick:</h4>
            <p>Accessible and iconic, this site is a must-visit for breathtaking views!</p>
            <div class="transport" data-from="ChIJ5Y6l4Q0Z2jERYL0KDIjT6v0" data-to="ChIJBTYg1g4Z2jERp_MBbu5erWY">
                <strong>Transport Options:</strong>
                <div>Ride (9 mins) - SGD 6.82 - Carbon: 0.781 kg - Quick route via Grab/Taxi.</div>
                <div>Cycle (7 mins) - Free - Carbon: 0 kg - Eco-friendly cycling option.</div>
            </div>
        </div>
        <div class="place" data-place-id="ChIJu7ZXhhsW2jERIrsz6gV3oSE">
            <strong>Arulmigu Velmurugan Gnanamuneeswarar Temple</strong>
            <p>A serene Hindu temple known for its intricate architecture and spiritual atmosphere.</p>
            <p>Address: 50 Rivervale Cres, Singapore 545029</p>
            <h4>Why this place pick:</h4>
            <p>Highlighted for its cultural significance and beautiful architecture, it is family-friendly!</p>
        </div>
    </div>

    <div class="day">
        <h2>Day 2: 2025-06-02</h2>
        <h3>Morning:</h3>
        <div class="place" data-place-id="ChIJUzbaYws82jERXHKMRlUKIug">
            <strong>Changi Beach Park</strong>
            <p>A picturesque beach with lush palm trees, perfect for a morning stroll or picnic.</p>
            <p>Address: Nicoll Dr, Singapore 498991</p>
            <h4>Why this place pick:</h4>
            <p>Beautiful and accessible, this spot is ideal for enjoyable family activities.</p>
            <div class="transport" data-from="accommodation" data-to="ChIJUzbaYws82jERXHKMRlUKIug">
                <strong>Transport Options:</strong>
                <div>Ride (56 mins) - SGD 34.33 - Carbon: 9.848 kg - Fast route via Grab/Taxi.</div>
                <div>Public Transport (106 mins) - SGD 6.24 - Carbon: 3.323 kg - MRT East West Line, then Bus 9.</div>
            </div>
        </div>
        <div class="place" data-place-id="ChIJvXm-YLw92jERvHHDBKUOYm4">
            <strong>Changi T1 Viewing Mall L3</strong>
            <p>A unique mall inside Changi Airport offering panoramic runway views, perfect for families.</p>
            <p>Address: 80 Airport Blvd., Singapore 819642</p>
            <h4>Why this place pick:</h4>
            <p>Free to enter and enjoyable views make this a worthwhile stop.</p>
            <div class="transport" data-from="ChIJUzbaYws82jERXHKMRlUKIug" data-to="ChIJvXm-YLw92jERvHHDBKUOYm4">
                <strong>Transport Options:</strong>
                <div>Ride (21 mins) - SGD 12.74 - Carbon: 2.666 kg - Fast via Grab/Taxi.</div>
                <div>Bus (48 mins) - SGD 3.45 - Carbon: 1.583 kg - Take Bus 19, then Bus 53.</div>
            </div>
        </div>

        <h3>Lunch:</h3>
        <div class="place" data-place-id="ChIJb2JGY9Q92jERa0tLgJ_wy6k">
            <strong>Daruma Tavern Punggol</strong>
            <p>A cozy restaurant offering various dishes, ideal for family meals after a morning at the beach.</p>
            <p>Address: 654A Punggol Dr., #01-10, Singapore 821654</p>
            <h4>Why this place pick:</h4>
            <p>A relaxed atmosphere with delicious food makes this a perfect lunch stop!</p>
        </div>

        <h3>Afternoon:</h3>
        <div class="place" data-place-id="ChIJ483Qk9YX2jERA0VOQV7d1tY">
            <strong>Singapore Changi Airport</strong>
            <p>A world-renowned airport with shopping, dining, and unique attractions.</p>
            <p>Address: 60 Airport Blvd., Singapore 819643</p>
            <h4>Why this place pick:</h4>
            <p>Highly accessible with various points of interest, it’s a family-friendly hub!</p>
            <div class="transport" data-from="ChIJvXm-YLw92jERvHHDBKUOYm4" data-to="ChIJ483Qk9YX2jERA0VOQV7d1tY">
                <strong>Transport Options:</strong>
                <div>Ride (3 mins) - SGD 4.76 - Carbon: 0.206 kg - Fastest route via Grab/Taxi.</div>
                <div>Public Transport (6 mins) - SGD 0.97 - Carbon: 0.031 kg - Direct public transport option.</div>
            </div>
        </div>
        <div class="place" data-place-id="ChIJw3l-FL4X2jERw2pScvHQCbg">
            <strong>Jewel Changi Airport</strong>
            <p>A stunning shopping destination with lush indoor gardens and a waterfall.</p>
            <p>Address: Singapore</p>
            <h4>Why this place pick:</h4>
            <p>Perfect for some family bonding time, filled with shops and dining options!</p>
        </div>
    </div>

    <div class="day">
        <h2>Day 3: 2025-06-03</h2>
        <h3>Morning:</h3>
        <div class="place" data-place-id="ChIJby-HbsAi2jERcHb1yx-Ei0A">
            <strong>Xtreme SkatePark</strong>
            <p>A popular spot for skateboard enthusiasts of all levels.</p>
            <p>Address: E Coast Park Service Rd, Singapore</p>
            <h4>Why this place pick:</h4>
            <p>Great for kids looking for some excitement and fun outdoor activities!</p>
            <div class="transport" data-from="accommodation" data-to="ChIJby-HbsAi2jERcHb1yx-Ei0A">
                <strong>Transport Options:</strong>
                <div>Ride (42 mins) - SGD 26.72 - Carbon: 7.375 kg - Fastest route via Grab/Taxi.</div>
                <div>MRT (78 mins) - SGD 4.93 - Carbon: 0.602 kg - Take East West Line, then Thomson East Coast Line.</div>
            </div>
        </div>
        <div class="place" data-place-id="ChIJCQHB6vc82jERq6iFty4Fzo4">
            <strong>Changi Chapel & Museum</strong>
            <p>Documenting Singapore's WWII history, this site blends learning with contemplation.</p>
            <p>Address: 1000 Upper Changi Rd N, Singapore 507707</p>
            <h4>Why this place pick:</h4>
            <p>Accessible and enriching, it provides a deeper understanding of history!</p>
            <div class="transport" data-from="ChIJby-HbsAi2jERcHb1yx-Ei0A" data-to="ChIJCQHB6vc82jERq6iFty4Fzo4">
                <strong>Transport Options:</strong>
                <div>Ride (26 mins) - SGD 16.48 - Carbon: 3.945 kg - Fast via Grab/Taxi.</div>
                <div>Bus (56 mins) - SGD 2.05 - Carbon: 0.705 kg - Take Bus 48, then Bus 2.</div>
            </div>
        </div>

        <h3>Lunch:</h3>
        <div class="place" data-place-id="ChIJseQsTQ0Z2jERqpBTWF0Zf84">
            <strong>Maxwell Food Centre</strong>
            <p>A vibrant hawker center known for its delicious street food.</p>
            <p>Address: 1 Kadayanallur St, Singapore 069184</p>
            <h4>Why this place pick:</h4>
            <p>Budget-friendly with many choices, this is a perfect lunch spot!</p>
            <div class="transport" data-from="ChIJCQHB6vc82jERq6iFty4Fzo4" data-to="ChIJseQsTQ0Z2jERqpBTWF0Zf84">
                <strong>Transport Options:</strong>
                <div>Ride (35 mins) - SGD 22.52 - Carbon: 5.98 kg - Quick Grab/Taxi route.</div>
                <div>Public Transport (64 mins) - SGD 3.41 - Carbon: 1.558 kg - Take Bus 4, then MRT Downtown Line.</div>
            </div>
        </div>

        <h3>Afternoon:</h3>
        <div class="place" data-place-id="ChIJi4ozyMEi2jERfEjUDq6iNb8">
            <strong>Bedok Jetty</strong>
            <p>A scenic spot for a leisurely walk, cycling, or fishing.</p>
            <p>Address: East Coast Park Service Road, Singapore 449876</p>
            <h4>Why this place pick:</h4>
            <p>Highly rated for its serene atmosphere, it's a great family-friendly end to your trip!</p>
        </div>
    </div>

    <div class="summary">
        <h2>Plan Overview</h2>
        <p>- 3 days with Morning / Lunch / Afternoon slots.</p>
        <p>- Estimated adult ticket spend ≈ SGD 942.5.</p>
        <p>- Approx. travel distance ≈ 229.4 km.</p>
        <p>- Accessible stops counted: 14.</p>
        <p>- Interest terms covered include: airport, arulmigu, attraction, bay, beach, bedok, budget, by, centre, changi, chapel, daruma...</p>
    </div>

    <div class="action-items">
        <h2>Action Items</h2>
        <ul>
            <li>Confirm ticket bookings</li>
            <li>Check local weather forecast</li>
            <li>Pack comfortable shoes</li>
        </ul>
    </div>
</body>
</html>"""
    pdf_bytes = create_pdf_bytes_plain_from_html(human_text,
                                                 title="Your Complete Trip Guide")  # create_pdf_bytes(human_text, title="Final Itinerary (Human-readable)")
    # Save PDF bytes to a local file
    with open("trip_guide.pdf", "wb") as pdf_file:
        pdf_file.write(pdf_bytes)
