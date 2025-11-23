# Deployment Guide

## Local Development

### Quick Start (Recommended for First Time)

```bash
./start_ui.sh
```

This script will:
1. Check for required files
2. Install dependencies
3. Launch Streamlit server at `http://localhost:8501`

### Manual Start

```bash
# Install dependencies
pip install -r Simulations/requirements_ui.txt

# Run the app
cd Simulations
streamlit run app.py
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t org-threat-surface:latest .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/Simulations/results:/app/results \
  org-threat-surface:latest
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f simulator

# Stop services
docker-compose down
```

## Cloud Deployment

### Streamlit Cloud (Free Tier)

1. Push your repository to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Select your repo and set main file to `Simulations/app.py`
5. Deploy

### Heroku

```bash
# Create Procfile
echo "web: cd Simulations && streamlit run app.py --logger.level=error" > Procfile

# Create setup.sh
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableXsrfProtection = false
" > ~/.streamlit/config.toml
EOF

# Deploy
heroku create your-app-name
heroku buildpacks:add heroku/python
git push heroku main
```

### AWS (Using EC2)

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Clone repository and setup
git clone your-repo
cd org-threat-surface
python3 -m venv venv
source venv/bin/activate
pip install -r Simulations/requirements_ui.txt

# Run in background with nohup
nohup streamlit run Simulations/app.py --server.port 8501 &

# Or use systemd service
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit Org Simulator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/org-threat-surface
ExecStart=/home/ubuntu/org-threat-surface/venv/bin/streamlit run Simulations/app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable streamlit
sudo systemctl start streamlit
```

### Google Cloud Run

```bash
# Create cloudbuild.yaml
cat > cloudbuild.yaml << 'EOF'
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/org-simulator', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/org-simulator']
  - name: 'gcr.io/cloud-builders/gke-deploy'
    args: ['run', '--filename=k8s/', '--image=gcr.io/$PROJECT_ID/org-simulator', '--location=us-central1', '--cluster=simulator-cluster']

images:
  - 'gcr.io/$PROJECT_ID/org-simulator'
EOF

# Deploy
gcloud builds submit
```

## Production Considerations

### SSL/TLS

```bash
# Using nginx with Let's Encrypt
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Obtain certificate
sudo certbot certonly --standalone -d yourdomain.com

# Configure nginx
sudo tee /etc/nginx/sites-available/streamlit > /dev/null <<EOF
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
    }
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://\$server_name\$request_uri;
}
EOF

sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Authentication

To add basic authentication, modify the Streamlit app:

```python
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("Enter password:", type="password")
        if password == "your-secure-password":
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect password")
            return False
    return True

if not check_password():
    st.stop()

# Rest of app code...
```

### Rate Limiting

For multi-user deployments, consider rate limiting:

```python
import time
from collections import defaultdict

# In app.py
request_times = defaultdict.default_factory(list)

def rate_limit(user_id: str, max_requests: int = 10, window_seconds: int = 60):
    now = time.time()
    cutoff = now - window_seconds
    request_times[user_id] = [t for t in request_times[user_id] if t > cutoff]
    
    if len(request_times[user_id]) >= max_requests:
        return False
    
    request_times[user_id].append(now)
    return True

# Use in simulation runs
if not rate_limit(st.session_state.user_id):
    st.error("Rate limit exceeded. Please wait before running another simulation.")
    st.stop()
```

### Performance Tuning

```bash
# Increase Python garbage collection
export PYTHONGC=400

# Use gunicorn instead of built-in server (for reverse proxy)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "streamlit.web.cli:_main_run_hook"
```

## Monitoring

### Logs

```bash
# Docker
docker logs -f org-threat-surface-simulator

# Systemd
sudo journalctl -u streamlit -f

# Local
# Streamlit logs appear in terminal where app is running
```

### Health Checks

```bash
# Simple health check
curl http://localhost:8501/_stcore/health

# With authentication
curl -H "Authorization: Bearer $TOKEN" http://localhost:8501/_stcore/health
```

### Resource Monitoring

```bash
# Docker stats
docker stats org-threat-surface-simulator

# System monitoring
htop
top -p $(pgrep -f 'streamlit run')
```

## Scaling

### Horizontal Scaling (Multiple Instances)

With load balancer (nginx):

```nginx
upstream streamlit_servers {
    server 10.0.1.10:8501;
    server 10.0.1.11:8501;
    server 10.0.1.12:8501;
}

server {
    listen 80;
    location / {
        proxy_pass http://streamlit_servers;
    }
}
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: org-simulator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: org-simulator
  template:
    metadata:
      labels:
        app: org-simulator
    spec:
      containers:
      - name: streamlit
        image: org-threat-surface:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: org-simulator-service
spec:
  selector:
    app: org-simulator
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Troubleshooting

### App Crashes
- Check logs for errors
- Verify data files exist
- Increase memory allocation
- Check disk space (esp. for large simulations)

### Slow Performance
- Reduce default simulation parameters
- Use caching for repeated queries
- Consider async processing for batch mode
- Profile with `python -m cProfile`

### Memory Leaks
- Monitor with `memory_profiler`
- Clear session state periodically
- Limit dataframe sizes
- Use generators for large datasets

## Maintenance

### Backups

```bash
# Backup results
tar -czf results_backup_$(date +%s).tar.gz Simulations/results/

# Backup database (if using)
pg_dump mydatabase > backup_$(date +%s).sql
```

### Updates

```bash
# Update dependencies
pip install --upgrade -r Simulations/requirements_ui.txt

# Rebuild Docker image
docker build --no-cache -t org-threat-surface:latest .
docker-compose up -d --build
```

## Support

For issues or questions:
1. Check logs
2. Review this guide
3. Check GitHub issues
4. Create detailed bug report with:
   - Streamlit version
   - Python version
   - Simulation parameters
   - Full error traceback
