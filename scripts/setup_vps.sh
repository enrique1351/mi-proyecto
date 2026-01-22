#!/bin/bash
# Setup script for VPS deployment

set -e

echo "🚀 Setting up Quant Trading System on VPS"
echo "=========================================="

# Update system
echo "📦 Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "📚 Installing dependencies..."
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup Docker
echo "🐳 Configuring Docker..."
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Create directories
echo "📁 Creating directories..."
mkdir -p data/{db,logs,models,backups}
mkdir -p config

# Setup environment
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env with your credentials"
fi

# Setup systemd service
echo "🔧 Creating systemd service..."
cat > /tmp/trading-system.service <<EOF
[Unit]
Description=Quant Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/trading-system.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo "✅ VPS setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with credentials"
echo "2. Enable service: sudo systemctl enable trading-system"
echo "3. Start service: sudo systemctl start trading-system"
echo "4. Check logs: sudo journalctl -u trading-system -f"
echo ""
