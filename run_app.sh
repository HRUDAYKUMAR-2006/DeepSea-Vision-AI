#!/bin/bash
echo "==============================================================================="
echo "    DEEPSEA VISION AI - MARINE SEARCH PARTY & UNDERWATER SURVEY SYSTEM"
echo "==============================================================================="
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "Launching application server on http://127.0.0.1:5000..."
python3 demo/app.py &
sleep 2

if which xdg-open > /dev/null; then
    xdg-open "http://127.0.0.1:5000"
elif which open > /dev/null; then
    open "http://127.0.0.1:5000"
fi
wait
