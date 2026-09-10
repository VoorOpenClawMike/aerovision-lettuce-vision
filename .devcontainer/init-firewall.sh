#!/bin/bash
set -euo pipefail
echo "[firewall] Setting up network allowlist..."
iptables -F; iptables -X; ipset destroy allowed-domains 2>/dev/null || true
iptables -P INPUT DROP; iptables -P FORWARD DROP; iptables -P OUTPUT DROP
iptables -A INPUT -i lo -j ACCEPT; iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT; iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT -p udp --sport 53 -j ACCEPT; iptables -A INPUT -p tcp --sport 53 -j ACCEPT
ipset create allowed-domains hash:ip
ALLOWED_DOMAINS=(
  "api.anthropic.com" "console.anthropic.com" "claude.ai"
  "github.com" "codeload.github.com" "objects.githubusercontent.com" "raw.githubusercontent.com"
  "registry.npmjs.org" "pypi.org" "files.pythonhosted.org" "download.pytorch.org"
)
for domain in "${ALLOWED_DOMAINS[@]}"; do
  ips=$(dig +short "$domain" A | grep -E '^[0-9]+\.' || true)
  for ip in $ips; do ipset add allowed-domains "$ip" 2>/dev/null || true; done
done
iptables -A OUTPUT -p tcp --dport 443 -m set --match-set allowed-domains dst -j ACCEPT
iptables -A OUTPUT -p tcp --dport 80 -m set --match-set allowed-domains dst -j ACCEPT
echo "[firewall] Active. Outbound restricted to: ${ALLOWED_DOMAINS[*]}"
