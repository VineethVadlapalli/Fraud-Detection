import requests
import time
import random
from datetime import datetime

# Make sure this matches your AWS IP and the /detect endpoint
URL = "http://13.61.22.182:8000/detect"

def generate_fake_transaction():
    """Generates a payload matching your specific Pydantic schema."""
    return {
        "amount": round(random.uniform(5.0, 5000.0), 2),
        "location_lat": round(random.uniform(40.0, 41.0), 4),
        "location_lon": round(random.uniform(-74.5, -73.5), 4),
        "merchant_id": f"merchant_{random.randint(100, 999)}",
        "timestamp": datetime.now().isoformat(),
        "transaction_id": f"txn_{random.randint(10000, 99999)}",
        "user_id": f"user_{random.randint(100, 999)}"
    }

def run_stress_test(total_requests=100):
    print(f"🚀 Starting Stress Test: Sending {total_requests} requests to AWS...")
    success_count = 0
    start_time = time.time()

    for i in range(total_requests):
        payload = generate_fake_transaction()
        try:
            # We use json=payload to send it as a JSON body
            response = requests.post(URL, json=payload, timeout=10)
            if response.status_code == 200:
                success_count += 1
                result = response.json()
                # Assuming your API returns 'fraud_score' or similar
                score = result.get('anomaly_score', 'N/A')
                print(f"[{i+1}] Success! Score: {score} | Status: {response.status_code}")
            else:
                print(f"[{i+1}] Failed! Status: {response.status_code} | Error: {response.text}")
        except Exception as e:
            print(f"[{i+1}] Connection Error: {e}")
        
        # 0.1s delay = 10 requests per second
        time.sleep(0.1)

    duration = time.time() - start_time
    print(f"\n✅ Done! Success Rate: {success_count}/{total_requests}")
    print(f"Total Time: {duration:.2f}s | Avg: {duration/total_requests:.3f}s/req")

if __name__ == "__main__":
    run_stress_test(100)