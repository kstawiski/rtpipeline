# Security Guide - rtpipeline

This document provides security guidance for deploying rtpipeline in production environments.

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Default Configuration Security](#default-configuration-security)
3. [Production Deployment Security](#production-deployment-security)
4. [Authentication & Authorization](#authentication--authorization)
5. [Network Security](#network-security)
6. [Data Protection](#data-protection)
7. [Container Security](#container-security)
8. [Monitoring & Incident Response](#monitoring--incident-response)
9. [HIPAA Compliance Considerations](#hipaa-compliance-considerations)
10. [Security Checklist](#security-checklist)

---

## Security Overview

### Design Philosophy

rtpipeline is designed with security in mind:

- ✅ **Container isolation**: Runs in Docker containers with minimal privileges
- ✅ **Non-root user**: Operates as UID 1000 (non-root)
- ✅ **Capability dropping**: Only essential Linux capabilities enabled
- ✅ **Path traversal protection**: Prevents directory traversal attacks
- ✅ **Input validation**: File type and size validation
- ✅ **Secure file handling**: Uses `secure_filename()` for uploads

### Security Model

```
┌─────────────────────────────────────────────┐
│  Network Layer (Your Responsibility)        │
│  ├─ Firewall                                │
│  ├─ HTTPS/TLS                               │
│  └─ Authentication (Reverse Proxy)          │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  rtpipeline Web UI (Port 8080)              │
│  ├─ Input Validation                        │
│  ├─ Path Traversal Protection               │
│  └─ Secure File Upload                      │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Docker Container (Security Hardened)       │
│  ├─ Non-root user (UID 1000)                │
│  ├─ Minimal capabilities                    │
│  ├─ Read-only code mount                    │
│  └─ Resource limits                         │
└─────────────────────────────────────────────┘
```

---

## Default Configuration Security

### ⚠️ Important: Default Setup is for Local/Trusted Use Only

The default `docker-compose.yml` configuration is designed for:

- ✅ **Local development** on a single workstation
- ✅ **Trusted network** environments (hospital intranet)
- ✅ **Research use** on isolated systems
- ❌ **NOT for internet-facing deployments** without additional security

### Default Security Features

1. **No Authentication**
   - Web UI has no built-in authentication
   - Anyone with network access to port 8080 can upload files and submit jobs
   - **Mitigation**: Only expose to trusted networks

2. **HTTP Only (No HTTPS)**
   - Communication is unencrypted
   - **Risk**: Data can be intercepted on the network
   - **Mitigation**: Use in trusted networks or add reverse proxy with TLS

3. **Secret Key**
   - Flask generates random secret key if not provided
   - **Impact**: Sessions are invalidated on container restart
   - **Mitigation**: Set `SECRET_KEY` environment variable for persistence

---

## Production Deployment Security

### Minimum Production Security Requirements

1. ✅ **Authentication** (reverse proxy or API token)
2. ✅ **HTTPS/TLS** encryption
3. ✅ **Firewall** rules
4. ✅ **Audit logging**
5. ✅ **Regular updates**

### Architecture Options

#### Option 1: Reverse Proxy (Recommended)

```
Internet → [Firewall] → [nginx + HTTPS + Auth] → rtpipeline (localhost:8080)
```

**Benefits**:
- Industry-standard approach
- Mature authentication options
- TLS termination
- Rate limiting
- WAF integration possible

#### Option 2: VPN Access Only

```
Internet → [VPN Gateway] → [Private Network] → rtpipeline (port 8080)
```

**Benefits**:
- Simple to implement
- Leverages existing VPN infrastructure
- Network-level isolation

#### Option 3: Internal Network Only

```
[Hospital Intranet] → rtpipeline (no internet access)
```

**Benefits**:
- Maximum isolation
- Suitable for air-gapped environments
- Compliance-friendly

---

## Authentication & Authorization

### Option A: Nginx Reverse Proxy with Basic Auth (Simplest)

**1. Create password file:**

```bash
sudo apt-get install apache2-utils
htpasswd -c /etc/nginx/.htpasswd rtuser
```

**2. Configure nginx:**

```nginx
# /etc/nginx/sites-available/rtpipeline
server {
    listen 443 ssl http2;
    server_name rtpipeline.yourdomain.com;

    # TLS configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Authentication
    auth_basic "rtpipeline Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Rate limiting
    limit_req zone=rtpipeline_limit burst=20 nodelay;

    # Reverse proxy
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Upload size (must match rtpipeline limit)
        client_max_body_size 50G;

        # Timeouts for long-running uploads/processing
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}

# HTTP redirect
server {
    listen 80;
    server_name rtpipeline.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

**3. Rate limiting configuration:**

```nginx
# In http block of /etc/nginx/nginx.conf
http {
    limit_req_zone $binary_remote_addr zone=rtpipeline_limit:10m rate=10r/s;
    # ...
}
```

**4. Enable and restart:**

```bash
sudo ln -s /etc/nginx/sites-available/rtpipeline /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option B: API Token Authentication (Advanced)

**Modify docker-compose.yml:**

```yaml
environment:
  - API_TOKEN=your-secure-random-token-here
```

**Generate secure token:**

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Client usage:**

```bash
curl -H "Authorization: Bearer your-secure-random-token-here" \
     -F "files[]=@data.zip" \
     https://rtpipeline.yourdomain.com/api/upload
```

⚠️ **Note**: This requires code modifications to webui/app.py (not currently implemented).

### Option C: OAuth2/OIDC (Enterprise)

For enterprise deployments, consider:

- [oauth2-proxy](https://github.com/oauth2-proxy/oauth2-proxy) with Google/Azure AD
- Keycloak integration
- Hospital SSO integration

**Example with oauth2-proxy:**

```yaml
services:
  oauth2-proxy:
    image: quay.io/oauth2-proxy/oauth2-proxy:latest
    command:
      - --provider=google
      - --client-id=YOUR_CLIENT_ID
      - --client-secret=YOUR_CLIENT_SECRET
      - --upstream=http://rtpipeline:8080
      - --http-address=0.0.0.0:4180
      - --cookie-secret=YOUR_COOKIE_SECRET
    ports:
      - "443:4180"
```

---

## Network Security

### Firewall Configuration

**1. Ubuntu/Debian (ufw):**

```bash
# Allow SSH (if managing remotely)
sudo ufw allow 22/tcp

# Allow HTTPS only (if using reverse proxy)
sudo ufw allow 443/tcp

# DO NOT expose rtpipeline directly
# sudo ufw allow 8080/tcp  # ❌ NEVER DO THIS

# Enable firewall
sudo ufw enable
```

**2. CentOS/RHEL (firewalld):**

```bash
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

**3. Docker-specific:**

Update docker-compose.yml to bind to localhost only:

```yaml
services:
  rtpipeline:
    ports:
      - "127.0.0.1:8080:8080"  # Only accessible from localhost
```

### Network Segmentation

For hospital/enterprise environments:

```
┌──────────────────────────────────────┐
│  DMZ (Public-facing)                 │
│  └─ Reverse Proxy (nginx)            │
└──────────────────────────────────────┘
                ↓ (Firewall)
┌──────────────────────────────────────┐
│  Application Tier                    │
│  └─ rtpipeline Container              │
└──────────────────────────────────────┘
                ↓ (Firewall)
┌──────────────────────────────────────┐
│  Data Tier                           │
│  └─ Persistent Storage (NFS/NAS)     │
└──────────────────────────────────────┘
```

---

## Data Protection

### Data at Rest

**1. Encrypt storage volumes:**

```bash
# LUKS encryption for Linux
sudo cryptsetup luksFormat /dev/sdb
sudo cryptsetup open /dev/sdb rtpipeline_data
sudo mkfs.ext4 /dev/mapper/rtpipeline_data
```

**2. Docker volume encryption:**

```yaml
volumes:
  output:
    driver: local
    driver_opts:
      type: none
      device: /encrypted/path
      o: bind
```

### Data in Transit

**1. Always use HTTPS** for web access (see nginx config above)

**2. VPN for administrative access:**

```bash
# Example: WireGuard VPN
sudo apt install wireguard
# Configure WireGuard...
```

### Data Retention

**1. Automatic cleanup:**

```bash
# Cron job to delete old jobs after 30 days
0 2 * * * find /data/output -type d -mtime +30 -exec rm -rf {} \;
```

**2. Audit logging:**

```yaml
services:
  rtpipeline:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Anonymization

Use the built-in anonymization script for de-identification:

```bash
python scripts/anonymize_pipeline_results.py \
  --input Data_Snakemake \
  --output Data_Snakemake_anonymized \
  --overwrite --verbose
```

**Features**:
- Rewrites patient identifiers
- Anonymizes DICOM headers
- Generates re-identification key (store securely!)

---

## Container Security

### Security Hardening Checklist

The default Docker configuration already includes:

- ✅ Non-root user (UID 1000)
- ✅ `no-new-privileges:true`
- ✅ Minimal capabilities (drops ALL, adds only required)
- ✅ Read-only code mounts
- ✅ Resource limits (CPU, memory)

### Additional Hardening

**1. Scan Docker image for vulnerabilities:**

```bash
# Using Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image kstawiski/rtpipeline:latest

# Using Snyk
snyk container test kstawiski/rtpipeline:latest
```

**2. Enable Docker Content Trust:**

```bash
export DOCKER_CONTENT_TRUST=1
```

**3. AppArmor/SELinux profiles:**

```yaml
services:
  rtpipeline:
    security_opt:
      - apparmor=docker-default
      - no-new-privileges:true
```

**4. Rootless Docker (advanced):**

```bash
# Run Docker daemon as non-root user
dockerd-rootless.sh
```

---

## Monitoring & Incident Response

### Security Logging

**1. Enable audit logging:**

```yaml
services:
  rtpipeline:
    logging:
      driver: syslog
      options:
        syslog-address: "tcp://your-syslog-server:514"
        tag: "rtpipeline"
```

**2. Monitor for suspicious activity:**

```bash
# Watch for failed authentication attempts (if using basic auth)
tail -f /var/log/nginx/error.log | grep "auth"

# Monitor file uploads
tail -f /data/logs/*.log | grep "upload"

# Watch container logs
docker logs -f rtpipeline
```

### Incident Response Plan

**1. Suspected compromise:**

```bash
# Immediate actions
docker-compose down          # Stop container
docker ps -a                  # Check for other containers
sudo iptables -L              # Verify firewall rules
find /data -mtime -1          # Check recent file modifications

# Preserve evidence
docker logs rtpipeline > incident_$(date +%Y%m%d).log
tar czf evidence_$(date +%Y%m%d).tar.gz /data/logs
```

**2. Contact information:**

- IT Security Team: [your-security-team@hospital.com]
- System Administrator: [admin@hospital.com]
- Vendor Support: [GitHub Issues](https://github.com/kstawiski/rtpipeline/issues)

---

## HIPAA Compliance Considerations

⚠️ **Disclaimer**: rtpipeline is a research tool. HIPAA compliance is the responsibility of the deploying institution.

### Technical Safeguards

#### Access Controls (Required)

- ✅ **Unique user identification**: Implement user authentication
- ✅ **Emergency access**: Document break-glass procedures
- ✅ **Automatic logoff**: Configure session timeouts
- ✅ **Encryption**: Use HTTPS + volume encryption

#### Audit Controls (Required)

```yaml
# Enhanced logging for HIPAA
services:
  rtpipeline:
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "10"
        labels: "hipaa-audit"
    environment:
      - LOG_LEVEL=INFO  # Capture all API calls
```

#### Integrity Controls (Addressable)

- ✅ **Checksum verification**: Verify uploaded DICOM integrity
- ✅ **Digital signatures**: Consider signing output files
- ✅ **Version control**: Track configuration changes

#### Transmission Security (Addressable)

- ✅ **Encryption in transit**: HTTPS/TLS 1.2+
- ✅ **Integrity controls**: Checksums for data transfer

### Administrative Safeguards

**1. Security risk analysis:**
- Document threat model
- Perform annual security reviews

**2. Workforce training:**
- Train staff on secure upload procedures
- Document incident response procedures

**3. Business associate agreements:**
- If using cloud hosting, execute BAA with provider

### Physical Safeguards

- ✅ **Facility access controls**: Secure server room
- ✅ **Workstation security**: Lock computers when unattended
- ✅ **Device and media controls**: Encrypt backup drives

### Compliance Documentation

Maintain:

1. **Security policies and procedures**
2. **Access control lists** (who has access)
3. **Audit logs** (retained for 6 years)
4. **Incident response logs**
5. **Training records**

---

## Security Checklist

### Pre-Deployment

- [ ] Security risk assessment completed
- [ ] Network architecture reviewed
- [ ] Firewall rules configured
- [ ] HTTPS/TLS certificates obtained
- [ ] Authentication method selected and tested
- [ ] Logging and monitoring configured
- [ ] Incident response plan documented
- [ ] Staff trained on security procedures

### Deployment

- [ ] Docker image scanned for vulnerabilities
- [ ] Container runs as non-root user
- [ ] Reverse proxy configured with TLS
- [ ] Authentication enforced
- [ ] Firewall rules active
- [ ] Port 8080 NOT exposed to internet
- [ ] Volume encryption enabled (if required)
- [ ] Audit logging active

### Post-Deployment

- [ ] Monitor logs daily for anomalies
- [ ] Review access logs weekly
- [ ] Update Docker images monthly
- [ ] Scan for vulnerabilities quarterly
- [ ] Review security policies annually
- [ ] Test incident response procedures annually
- [ ] Audit user access quarterly

### Data Handling

- [ ] DICOM sources validated/trusted
- [ ] PHI minimization implemented (anonymization)
- [ ] Retention policies enforced
- [ ] Backup encryption verified
- [ ] Disposal procedures documented

---

## Quick Start: Secure Production Setup

**1. Install nginx and certbot:**

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

**2. Get TLS certificate:**

```bash
sudo certbot --nginx -d rtpipeline.yourdomain.com
```

**3. Configure basic authentication:**

```bash
sudo htpasswd -c /etc/nginx/.htpasswd rtuser
```

**4. Update docker-compose.yml:**

```yaml
services:
  rtpipeline:
    ports:
      - "127.0.0.1:8080:8080"  # Localhost only
```

**5. Copy nginx config:**

```bash
sudo cp docs/nginx-rtpipeline.conf /etc/nginx/sites-available/rtpipeline
sudo ln -s /etc/nginx/sites-available/rtpipeline /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

**6. Start rtpipeline:**

```bash
docker-compose up -d
```

**7. Test:**

```bash
curl -u rtuser:password https://rtpipeline.yourdomain.com/health
```

---

## Additional Resources

- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST SP 800-190: Application Container Security](https://csrc.nist.gov/publications/detail/sp/800-190/final)
- [HHS HIPAA Security Rule Guidance](https://www.hhs.gov/hipaa/for-professionals/security/index.html)

---

## Reporting Security Issues

If you discover a security vulnerability in rtpipeline:

1. **DO NOT** open a public GitHub issue
2. Email the maintainer directly (see repository for contact)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

Response time: 72 hours for acknowledgment, 30 days for fix (critical issues)

---

**Last Updated**: 2025-11-19
**Version**: 1.0
