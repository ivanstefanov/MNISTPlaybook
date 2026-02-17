# Kubeflow Installation on Windows WSL2 - Complete Guide

This document captures a successful Kubeflow installation on Windows using WSL2, including all tool versions, system configuration, and troubleshooting steps encountered.

## Table of Contents

- [System Configuration](#system-configuration)
- [Prerequisites](#prerequisites)
- [Tool Versions Used](#tool-versions-used)
- [Installation Steps](#installation-steps)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Accessing Kubeflow](#accessing-kubeflow)

---

## System Configuration

### WSL2 Configuration

- **WSL Version**: WSL2
- **Kernel**: Linux 6.6.87.2-microsoft-standard-WSL2
- **Distribution**: Ubuntu (Debian-based)
- **Hostname**: NTB1199

### Resource Allocation

- **RAM**: 16 GB (16377024 kB total)
- **CPU Cores**: 12 cores
- **Disk Space**: 1 TB available (1007G total, 953G free)

### Kernel Parameters (Pre-configured)

These were already set correctly for this installation:

```bash
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
```

**Note**: The Kubeflow README recommends:
- `fs.inotify.max_user_instances = 2280` (minimum)
- `fs.inotify.max_user_watches = 1255360` (minimum)

Our values exceed these minimums, which is fine.

If you need to set these, run:
```bash
sudo sysctl fs.inotify.max_user_instances=2280
sudo sysctl fs.inotify.max_user_watches=1255360
```

---

## Prerequisites

### Windows Requirements

1. **Docker Desktop for Windows**
   - Version: 29.1.3 (or later)
   - WSL2 backend enabled
   - WSL Integration enabled for your distribution

2. **WSL2 Enabled**
   - Run in PowerShell as Administrator:
     ```powershell
     wsl --set-default-version 2
     ```

### Linux Tools (Installed in WSL)

All tools listed below with their versions in the next section.

---

## Tool Versions Used

This installation was successful with the following versions:

| Tool                  | Version          | Notes                                     |
| --------------------- | ---------------- | ----------------------------------------- |
| **Kind**              | v0.31.0          | Kubernetes in Docker (requirement: 0.27+) |
| **kubectl**           | v1.35.0          | Kubernetes CLI                            |
| **Kustomize**         | v5.7.1           | Required exact version for Kubeflow       |
| **Docker**            | 29.1.3           | Docker Desktop for Windows                |
| **Kubernetes**        | v1.34.0          | Running in Kind cluster                   |
| **Container Runtime** | containerd 2.1.3 | Used by Kind                              |

---

## Installation Steps

### Step 1: Enable Docker Desktop WSL Integration

**On Windows (Docker Desktop):**

1. Open Docker Desktop
2. Go to **Settings** → **Resources** → **WSL Integration**
3. Enable "Enable integration with my default WSL distro"
4. Toggle on your specific WSL distribution
5. Click **Apply & Restart**

**⚠️ Important**: After Docker Desktop restarts, you may need to restart WSL for docker group permissions to take effect. See [Troubleshooting](#docker-permission-issues) below.

### Step 2: Install Kind

Kind should already be installed. Verify with:
```bash
kind version
```

If not installed, download from: https://kind.sigs.k8s.io/docs/user/quick-start/#installation

### Step 3: Install kubectl

```bash
# Download kubectl v1.35.0 (or latest stable)
cd /tmp
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install to user local bin
mkdir -p ~/.local/bin
chmod +x kubectl
mv kubectl ~/.local/bin/

# Add to PATH (if not already added)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
kubectl version --client
```

### Step 4: Install Kustomize v5.7.1

**⚠️ Critical**: Kubeflow requires exactly version 5.7.1

```bash
# Download kustomize 5.7.1
cd /tmp
curl -Lo kustomize.tar.gz https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.7.1/kustomize_v5.7.1_linux_amd64.tar.gz

# Extract and install
tar -xzf kustomize.tar.gz
chmod +x kustomize
mv kustomize ~/.local/bin/

# Clean up
rm kustomize.tar.gz

# Verify installation
kustomize version
```

### Step 5: Verify Docker Access

```bash
# Test docker works without sudo
docker ps

# If you get permission denied, see Troubleshooting section below
```

### Step 6: Create Kind Cluster

```bash
# Create cluster with Kubeflow configuration
cat <<'EOF' | kind create cluster --name=kubeflow --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest/node:v1.34.0@sha256:7416a61b42b1662ca6ca89f02028ac133a309a2a30ba309614e8ec94d976dc5a
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "https://kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
EOF
```

**Expected output:**
```
Creating cluster "kubeflow" ...
 ✓ Ensuring node image (kindest/node:v1.34.0) 🖼
 ✓ Preparing nodes 📦
 ✓ Writing configuration 📜
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-kubeflow"
```

### Step 7: Save Kubeconfig

```bash
# Save kubeconfig to temporary location
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config

# Set KUBECONFIG environment variable
export KUBECONFIG=/tmp/kubeflow-config

# Verify cluster connection
kubectl cluster-info
```

### Step 8: Create Docker Registry Secret

```bash
# Ensure you're logged into Docker Hub
docker login

# Create secret for pulling images
kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson
```

### Step 9: Install Kubeflow

**Method 1: Single Command with Retry (Recommended)**
1. Create the `~/tools/manifests` and clone the cubeflow manifest
```bash
mkdir -p ~/tools
cd ~/tools

# uncomment to remove the manifest if it still exists
# rm -rf manifests

git clone https://github.com/kubeflow/manifests.git
cd manifests

# sanity check: we need example/
ls -la example/kustomization.yaml
```

2. Create a retry script to handle webhook timing issues:

```bash
# Create installation script
cat > /tmp/install-kubeflow.sh <<'EOF'
#!/bin/bash
export KUBECONFIG=/tmp/kubeflow-config
cd ~/tools/manifests  # Adjust to your manifests repo location

count=0
max_retries=15

while [ $count -lt $max_retries ]; do
  echo "=== Installation attempt $((count + 1))/$max_retries ==="
  if kustomize build example | kubectl apply --server-side --force-conflicts -f - 2>&1; then
    echo "=== Installation successful! ==="
    exit 0
  fi
  echo "Retrying in 20 seconds..."
  sleep 20
  count=$((count + 1))
done

echo "=== Max retries reached ==="
exit 1
EOF

# Make executable and run
chmod +x /tmp/install-kubeflow.sh
/tmp/install-kubeflow.sh 2>&1 | tee /tmp/kubeflow-install.log
```

**Method 2: Manual Retry**

```bash
export KUBECONFIG=/tmp/kubeflow-config

# Run install command
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
  echo "Retrying to apply resources";
  sleep 20;
done
```

**Installation Duration**: Approximately 3-5 minutes with 2-3 retry attempts.

**Common warnings (safe to ignore):**
- `Warning: 'vars' is deprecated`
- `Warning: 'bases' is deprecated`
- `Warning: 'commonLabels' is deprecated`
- `Warning: unrecognized format "int32/int64/double/binary"`

**Expected errors on first attempt:**
```
Error from server (InternalError): failed calling webhook "webhook.serving.knative.dev"
Error from server (InternalError): failed calling webhook "clusterservingruntime.kserve-webhook-server.validator"
```

These are **normal** - webhooks aren't ready yet. The retry loop handles this automatically.

---

## Troubleshooting

### Docker Permission Issues

**Problem**: `docker ps` returns "permission denied"

**Cause**: After Docker Desktop enables WSL integration, your user session needs to refresh to recognize the docker group membership.

**Solutions** (choose one):

#### Option 1: Restart WSL (Recommended - Clean Fix)

On **Windows PowerShell**:
```powershell
wsl --shutdown
```

Then reopen WSL. Docker should work without sudo.

#### Option 2: Use newgrp (Quick - Current Session Only)

In your current WSL terminal:
```bash
newgrp docker
```

This creates a new shell with docker group active.

#### Option 3: Open New Terminal

Simply open a new WSL terminal window. The new session will have correct permissions.

**Verify fix:**
```bash
docker ps
# Should show container list without errors
```

### Webhook Connection Errors During Installation

**Problem**: Errors like:
```
Error from server (InternalError): failed calling webhook
```

**Cause**: Webhooks aren't ready when CRDs are being created (timing issue).

**Solution**: This is expected! The retry loop handles it. Just wait for retries.

**Timeframe**: Usually resolves within 1-3 retry attempts (20-60 seconds).

### metadata-grpc Pod CrashLoopBackOff

**Problem**: After installation, `metadata-grpc-deployment` pod shows CrashLoopBackOff.

**Logs show:**
```
MySQL database was not initialized. Please ensure your MySQL server is running
```

**Cause**: MySQL takes time to fully initialize. The metadata pod tries to connect before MySQL is ready.

**Solution**: Wait. Kubernetes will automatically retry.

**Recovery Time**: Usually recovers within 5-10 minutes (after 5-10 restarts).

**Verification:**
```bash
export KUBECONFIG=/tmp/kubeflow-config
kubectl get pods -n kubeflow | grep metadata-grpc
```

Look for:
```
metadata-grpc-deployment-xxx   2/2   Running   7 (6m ago)   17m
```

Status "2/2 Running" means it recovered (the "7" restarts is normal).

### Port Forward Issues

**Problem**: `kubectl port-forward` command doesn't work or connection refused.

**Cause**: KUBECONFIG not set or pods not ready.

**Solution**:
```bash
# Always set KUBECONFIG first
export KUBECONFIG=/tmp/kubeflow-config

# Wait for all pods to be ready
kubectl get pods -n istio-system
kubectl get pods -n kubeflow

# Then try port-forward
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

### Disk Space Issues

**Problem**: Installation fails with disk space errors.

**Cause**: Kind downloads large container images (~10-15GB total).

**Solution**: Ensure at least 50GB free space in WSL.

**Check space:**
```bash
df -h /
```

---

## Verification

### Step 1: Check All Namespaces

```bash
export KUBECONFIG=/tmp/kubeflow-config

# Check cert-manager
kubectl get pods -n cert-manager

# Check Istio
kubectl get pods -n istio-system

# Check Dex (Auth)
kubectl get pods -n auth

# Check OAuth2-Proxy
kubectl get pods -n oauth2-proxy

# Check Knative
kubectl get pods -n knative-serving

# Check Kubeflow components
kubectl get pods -n kubeflow

# Check Kubeflow system
kubectl get pods -n kubeflow-system

# Check user namespace
kubectl get pods -n kubeflow-user-example-com
```

### Step 2: Verify All Pods Running

```bash
export KUBECONFIG=/tmp/kubeflow-config

# Count Running pods in kubeflow namespace
kubectl get pods -n kubeflow --no-headers | awk '{print $3}' | grep -c Running
```

**Expected**: 34 pods Running (after metadata-grpc recovers)

### Step 3: Full Pod List

Expected pods in `kubeflow` namespace:
- admission-webhook-deployment
- cache-server
- centraldashboard
- jupyter-web-app-deployment
- katib-controller, katib-db-manager, katib-mysql, katib-ui
- kserve-controller-manager, kserve-localmodel-controller-manager, kserve-models-web-app
- kubeflow-pipelines-profile-controller
- metacontroller
- metadata-envoy-deployment, metadata-grpc-deployment, metadata-writer
- ml-pipeline, ml-pipeline-persistenceagent, ml-pipeline-scheduledworkflow
- ml-pipeline-ui, ml-pipeline-viewer-crd, ml-pipeline-visualizationserver
- mysql
- notebook-controller-deployment
- profiles-deployment
- pvcviewer-controller-manager
- seaweedfs
- spark-operator-controller, spark-operator-webhook
- tensorboard-controller-deployment, tensorboards-web-app-deployment
- volumes-web-app-deployment
- workflow-controller

---

## Accessing Kubeflow

### Port Forward to Kubeflow Dashboard

**In WSL Terminal:**
```bash
export KUBECONFIG=/tmp/kubeflow-config
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

**Output:**
```
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

**Keep this terminal running** - Do not close it while accessing Kubeflow.

### Access from Windows Browser

1. Open your browser (Chrome, Edge, Firefox)
2. Navigate to: `http://localhost:8080`
3. You should see the Dex login screen

### Default Credentials

**⚠️ IMPORTANT**: These are default credentials for testing only. Change them for production use.

- **Email**: `user@example.com`
- **Password**: `12341234`

### First Login

After logging in, you should see:
- Kubeflow Central Dashboard
- Navigation menu on the left
- "Welcome to Kubeflow" message

### Available Features

From the dashboard, you can access:
- **Notebooks**: Jupyter notebook servers
- **Pipelines**: ML pipeline creation and execution
- **Experiments (AutoML)**: Katib hyperparameter tuning
- **Models**: KServe model serving and management
- **Volumes**: Persistent storage management
- **Tensorboards**: Visualization servers

---

## Persistence Notes

### KUBECONFIG Environment Variable

The `KUBECONFIG` environment variable is **not persistent** across terminal sessions.

**For every new terminal session**, run:
```bash
export KUBECONFIG=/tmp/kubeflow-config
```

**To make it persistent**, add to `~/.bashrc`:
```bash
echo 'export KUBECONFIG=/tmp/kubeflow-config' >> ~/.bashrc
source ~/.bashrc
```

### Kind Cluster Persistence

The Kind cluster **persists** across WSL restarts and Windows reboots.

**To list clusters:**
```bash
kind get clusters
```

**To delete the cluster:**
```bash
kind delete cluster --name kubeflow
```

**To recreate the cluster**, re-run the steps from [Create Kind Cluster](#step-6-create-kind-cluster).

### Docker Desktop Requirement

Kind requires Docker Desktop to be running. If you shut down Docker Desktop on Windows, the Kind cluster becomes inaccessible until Docker Desktop restarts.

---

## Common Commands Reference

### Cluster Management

```bash
# List clusters
kind get clusters

# Get cluster info
kubectl cluster-info

# View all nodes
kubectl get nodes

# Delete cluster
kind delete cluster --name kubeflow
```

### Pod Management

```bash
# Set KUBECONFIG (required for all kubectl commands)
export KUBECONFIG=/tmp/kubeflow-config

# Get all pods in all namespaces
kubectl get pods -A

# Get pods in specific namespace
kubectl get pods -n kubeflow

# Watch pods (auto-refresh)
kubectl get pods -n kubeflow -w

# Describe a pod (detailed info)
kubectl describe pod <pod-name> -n kubeflow

# View pod logs
kubectl logs <pod-name> -n kubeflow

# View logs for specific container in pod
kubectl logs <pod-name> -n kubeflow -c <container-name>
```

### Service Access

```bash
# List services
kubectl get svc -n istio-system

# Port forward to dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

# Port forward to specific service
kubectl port-forward svc/<service-name> -n <namespace> <local-port>:<service-port>
```

---

## Cleanup

### Stop Port Forwarding

Press `Ctrl+C` in the terminal running `kubectl port-forward`

### Delete Kubeflow Cluster

```bash
kind delete cluster --name kubeflow
```

This removes:
- The Kind cluster
- All Kubeflow components
- All deployed workloads

**Note**: This does NOT uninstall kubectl, kustomize, or kind.

### Uninstall Tools (Optional)

```bash
# Remove binaries
rm ~/.local/bin/kubectl
rm ~/.local/bin/kustomize

# Remove kubeconfig
rm /tmp/kubeflow-config

# Docker and Kind are typically kept for future use
```

---

## Production Deployment Considerations

This guide is for **development/testing** environments. For production:

1. **Change default credentials** - See Kubeflow README for password changes
2. **Use HTTPS** - Required for secure cookies (not localhost)
3. **Resource limits** - Adjust based on workload requirements
4. **Persistent storage** - Configure proper PVC storage class
5. **High availability** - Use multi-node clusters
6. **Monitoring** - Add Prometheus/Grafana
7. **Backup** - Regular backups of pipelines, notebooks, data

---

## Version Compatibility Matrix

| Component      | Version Tested | Requirement                 |
| -------------- | -------------- | --------------------------- |
| WSL            | WSL2           | WSL2 required               |
| Kubernetes     | 1.34.0         | Master branch targets 1.34+ |
| Docker Desktop | 29.1.3         | Latest stable recommended   |
| Kind           | 0.31.0         | 0.27+ required              |
| kubectl        | 1.35.0         | Compatible with K8s 1.34    |
| Kustomize      | 5.7.1          | **Exactly 5.7.1** required  |

---

## Additional Resources

- **Kubeflow Manifests Repository**: https://github.com/kubeflow/manifests
- **Official Kubeflow Documentation**: https://www.kubeflow.org/docs/
- **Kind Documentation**: https://kind.sigs.k8s.io/
- **WSL Documentation**: https://learn.microsoft.com/en-us/windows/wsl/
- **Docker Desktop WSL Integration**: https://docs.docker.com/desktop/wsl/

---

## Success Indicators

Your installation is successful when:

✅ All 34 pods in `kubeflow` namespace show status `Running`:
```bash
kubectl get pods -n kubeflow
# fast check
kubectl get pods -n kubeflow --no-headers | awk '{print $3}' | sort | uniq -c
```

✅ All pods in `istio-system`, `cert-manager`, `auth`, `oauth2-proxy`, `knative-serving` namespaces are `Running`
```bash
kubectl get pods -n istio-system
kubectl get pods -n cert-manager
kubectl get pods -n auth
kubectl get pods -n oauth2-proxy
kubectl get pods -n knative-serving
# fast check
kubectl get pods -A | egrep 'CrashLoopBackOff|Pending|Error|ImagePull'
```

✅ Port forwarding connects successfully
```bash
#check for gateway
kubectl get svc -n istio-system istio-ingressgateway
# the actual port forwarding
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

✅ Dashboard loads at `http://localhost:8080` in windows browser
✅ Login with default credentials works (user@example.com, 12341234)
✅ Central Dashboard displays properly (Kubeflow Central Dashboard, left menu with notebooks, pipelines, experiments, models..., no error baners)
```bash
kubectl get pods -n kubeflow | grep centraldashboard
# final check (should contains just the header)
kubectl get pods -A | awk 'NR==1 || $4!="Running"'
```


---

**Installation Date**: January 21, 2026
**Kubeflow Version**: Master branch (latest as of installation date)
**Installation Time**: ~15 minutes (including retries and pod startup)
