#!/bin/bash

echo "🗄️  Creating MedSafe Supabase Tables"
echo "======================================"
echo ""

# Check if SQL file exists
if [ ! -f "/Users/sujalalmeida/Downloads/medsafe/server/create_required_tables.sql" ]; then
    echo "❌ SQL file not found!"
    exit 1
fi

echo "📋 This script will show you the SQL to create the required tables."
echo ""
echo "⚠️  IMPORTANT: You need to run this SQL in your Supabase Dashboard"
echo ""
echo "🔗 Steps:"
echo "   1. Go to: https://app.supabase.com"
echo "   2. Select your MedSafe project"
echo "   3. Click 'SQL Editor' in the left sidebar"
echo "   4. Click 'New Query'"
echo "   5. Copy the SQL below and paste it there"
echo "   6. Click 'Run' (or press Cmd+Enter)"
echo ""
echo "======================================"
echo "📝 SQL TO COPY:"
echo "======================================"
echo ""

cat /Users/sujalalmeida/Downloads/medsafe/server/create_required_tables.sql

echo ""
echo "======================================"
echo ""
echo "✅ After running the SQL in Supabase:"
echo "   1. Restart your backend: cd server && python app.py"
echo "   2. Refresh your browser"
echo "   3. The error will be gone!"
echo ""
echo "🎯 Then test by submitting patient data from the Lab Dashboard."
echo ""
