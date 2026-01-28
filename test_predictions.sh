#!/bin/bash

echo "🧪 Testing Patient Data Submission & Display"
echo "============================================"
echo ""

# Test with high BP patient
echo "📋 Test 1: High BP Patient (Should be Hypertension)"
echo "---------------------------------------------------"
curl -s -X POST http://localhost:5001/lab/add_patient_data \
  -H "Content-Type: application/json" \
  -d '{
    "lab_label": "Lab A",
    "age": 62,
    "gender": "male",
    "blood_type": "O+",
    "discomfort_level": 7,
    "symptom_duration": 10,
    "glucose": 95,
    "bp_sys": 160,
    "bp_dia": 98,
    "cholesterol": 200,
    "bmi": 32.0,
    "smoker_status": "no",
    "heart_rate": 75,
    "family_history": "hypertension",
    "medication_use": "none"
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print(f\"Disease Type: {data['disease_type']}\"); print(f\"Risk Score: {data['risk_score']:.1%}\"); print(f\"Model Accuracy: {data['local_accuracy']:.1%}\")"

echo ""
echo "📋 Test 2: High Glucose Patient (Should be Diabetes)"
echo "----------------------------------------------------"
curl -s -X POST http://localhost:5001/lab/add_patient_data \
  -H "Content-Type: application/json" \
  -d '{
    "lab_label": "Lab A",
    "age": 55,
    "gender": "female",
    "blood_type": "A+",
    "discomfort_level": 5,
    "symptom_duration": 14,
    "glucose": 145,
    "bp_sys": 125,
    "bp_dia": 80,
    "cholesterol": 210,
    "bmi": 28.5,
    "smoker_status": "no",
    "heart_rate": 72,
    "family_history": "diabetes",
    "medication_use": "none"
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print(f\"Disease Type: {data['disease_type']}\"); print(f\"Risk Score: {data['risk_score']:.1%}\"); print(f\"Model Accuracy: {data['local_accuracy']:.1%}\")"

echo ""
echo "📋 Test 3: Healthy Patient (Should be Healthy)"
echo "----------------------------------------------"
curl -s -X POST http://localhost:5001/lab/add_patient_data \
  -H "Content-Type: application/json" \
  -d '{
    "lab_label": "Lab A",
    "age": 30,
    "gender": "male",
    "blood_type": "O+",
    "discomfort_level": 2,
    "symptom_duration": 3,
    "glucose": 90,
    "bp_sys": 115,
    "bp_dia": 75,
    "cholesterol": 180,
    "bmi": 23.0,
    "smoker_status": "no",
    "heart_rate": 70,
    "family_history": "none",
    "medication_use": "none"
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print(f\"Disease Type: {data['disease_type']}\"); print(f\"Risk Score: {data['risk_score']:.1%}\"); print(f\"Model Accuracy: {data['local_accuracy']:.1%}\")"

echo ""
echo "============================================"
echo "✅ If all three show different results, backend is working!"
echo "🌐 If frontend still shows wrong values, it's a display issue."
echo ""
echo "💡 Try:"
echo "   1. Open browser DevTools (F12)"
echo "   2. Go to Console tab"
echo "   3. Submit a patient"
echo "   4. Look for 'Backend Response' log"
echo "   5. Check if disease_type and risk_score are correct"
echo ""
