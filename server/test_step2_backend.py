#!/usr/bin/env python3
"""
Test script for Step 2 backend endpoints
Run this after starting the Flask server
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_get_global_model_info():
    print("\n=== Testing /lab/get_global_model_info ===")
    response = requests.get(f"{BASE_URL}/lab/get_global_model_info?lab_label=Lab A")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_download_global_model():
    print("\n=== Testing /lab/download_global_model ===")
    response = requests.post(
        f"{BASE_URL}/lab/download_global_model",
        json={"lab_label": "Lab A"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_get_round_history():
    print("\n=== Testing /admin/get_round_history ===")
    response = requests.get(f"{BASE_URL}/admin/get_round_history")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total rounds: {data.get('total_rounds')}")
    print(f"Convergence stats: {json.dumps(data.get('convergence_stats'), indent=2)}")
    if data.get('rounds'):
        print(f"Latest round: {json.dumps(data['rounds'][-1], indent=2)}")
    return data

def test_get_convergence_stats():
    print("\n=== Testing /admin/get_convergence_stats ===")
    response = requests.get(f"{BASE_URL}/admin/get_convergence_stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_aggregation_status():
    print("\n=== Testing /admin/get_aggregation_status ===")
    response = requests.get(f"{BASE_URL}/admin/get_aggregation_status")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Ready labs: {data.get('ready_labs')}/{data.get('total_labs')}")
    print(f"Current global model version: {data.get('current_global_model', {}).get('version')}")
    return data

if __name__ == "__main__":
    print("🚀 Testing Step 2 Backend Endpoints")
    print("=" * 60)
    
    try:
        # Test endpoints
        test_aggregation_status()
        test_get_global_model_info()
        test_get_round_history()
        test_get_convergence_stats()
        
        # Uncomment to test download (creates files)
        # test_download_global_model()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to Flask server")
        print("Make sure the server is running on http://localhost:5001")
    except Exception as e:
        print(f"\n❌ Error: {e}")
