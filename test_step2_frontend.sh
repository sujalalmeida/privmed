#!/bin/bash

echo "🧪 Testing Step 2 - Global Model Download Feature"
echo "=================================================="
echo ""

# Test 1: Check if backend is running
echo "1️⃣ Testing backend connectivity..."
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "   ✅ Backend is running on port 5001"
else
    echo "   ❌ Backend is NOT running!"
    echo "   👉 Start it with: cd server && python app.py"
    exit 1
fi

echo ""

# Test 2: Check global model info endpoint
echo "2️⃣ Testing /lab/get_global_model_info endpoint..."
RESPONSE=$(curl -s "http://localhost:5001/lab/get_global_model_info?lab_label=lab_sim")

if echo "$RESPONSE" | grep -q '"available": true'; then
    echo "   ✅ Global model is available!"
    echo "   📊 Model Info:"
    echo "$RESPONSE" | python3 -m json.tool | grep -A 5 "global_model"
else
    echo "   ⚠️  No global model available yet"
    echo "   💡 Create one by:"
    echo "      1. Submit patient data from Lab Dashboard"
    echo "      2. Send model update from Model Training tab"
    echo "      3. Aggregate models from Admin Dashboard"
fi

echo ""

# Test 3: Check if frontend is running
echo "3️⃣ Testing frontend connectivity..."
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "   ✅ Frontend is running on port 5173"
else
    echo "   ❌ Frontend is NOT running!"
    echo "   👉 Start it with: npm run dev"
    exit 1
fi

echo ""

# Test 4: Verify the fix
echo "4️⃣ Verifying the fix..."
if echo "$RESPONSE" | grep -q '"error"'; then
    echo "   ❌ Backend still returning errors"
    echo "   Error: $(echo "$RESPONSE" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("error", "Unknown"))')"
else
    echo "   ✅ Backend is working correctly!"
fi

echo ""
echo "=================================================="
echo "📋 Summary:"
echo ""
echo "To see the Global Model Download feature:"
echo "1. Open http://localhost:5173 in your browser"
echo "2. Login as a Lab user"
echo "3. Navigate to the 'Model Training' tab"
echo "4. You should see a purple/indigo card with:"
echo "   📥 'Global Model Available' heading"
echo "   🔄 Version comparison (Global vs Local)"
echo "   ⬇️  'Download Global Model' button"
echo "   🔄 Refresh button"
echo ""
echo "If you don't see it, check browser console (F12) for errors"
echo "The feature appears when available=true from backend API"
echo ""
