#!/bin/bash
# Auto-deployment script for Synapse
# Eliminates Woodpecker OAuth dependency by using systemd timer

set -e

echo "🚀 Starting auto-deployment..."

# Pull latest code from GitHub
cd /opt/synapse
sudo -u synapse git fetch origin main
sudo -u synapse git reset --hard origin/main

# Update dependencies
echo "📦 Updating Python dependencies..."
sudo -u synapse /opt/synapse/venv/bin/pip install -r requirements.txt

# Restart service
echo "🔄 Restarting Synapse service..."
sudo systemctl restart synapse-server

# Wait for service to start
sleep 5

# Health check
echo "🏥 Running health check..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Synapse service is healthy (HTTP $HTTP_CODE)"
else
    echo "❌ Synapse service failed to start (HTTP $HTTP_CODE)"
    sudo systemctl status synapse-server --no-pager -n 20
    exit 1
fi

echo "✅ Auto-deployment complete"
echo "🔗 Synapse available at:"
echo "   Local: http://localhost:8000"
echo "   Tailscale: http://ubuntuserver.tail70104d.ts.net:8000"
echo "   MCP: http://ubuntuserver.tail70104d.ts.net:8000/mcp"
