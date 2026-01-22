#!/bin/bash
# Setup script for Raspberry Pi deployment

set -e

echo "🚀 Setting up Quant Trading System on Raspberry Pi"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
echo "🐍 Installing Python and system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    redis-server \
    postgresql \
    postgresql-contrib

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup databases
echo "🗄️  Setting up databases..."
sudo systemctl enable postgresql redis-server
sudo systemctl start postgresql redis-server

# Create directories
echo "📁 Creating data directories..."
mkdir -p data/{db,logs,models,backups}

# Setup environment
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your credentials"
fi

echo "✅ Setup complete! Edit .env and run: python main.py"
