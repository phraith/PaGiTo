#!/bin/bash

vault auth enable kubernetes
vault secrets enable database

vault write auth/kubernetes/role/webapp \
        bound_service_account_names=vault \
        bound_service_account_namespaces=default \
        policies=webapp \
        ttl=24h


JWT=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
KUBERNETES_HOST=https://${KUBERNETES_PORT_443_TCP_ADDR}:443
KUBERNETES_CA_CERT=$(cat /var/run/secrets/kubernetes.io/serviceaccount/ca.crt)
vault write auth/kubernetes/config \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
    kubernetes_host=https://${KUBERNETES_PORT_443_TCP_ADDR}:443 \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt disable_iss_validation=true \
    issuer="https://kubernetes.default.svc.cluster.local"



vault write auth/kubernetes/role/internal-app \
    bound_service_account_names=internal-app \
    bound_service_account_namespaces=default \
    policies=internal-app \
    ttl=24h


vault policy write webapp - <<EOF
path "database/creds/postgres" {
  capabilities = ["read"]
}
EOF

vault write auth/kubernetes/role/webapp \
        bound_service_account_names=vault \
        bound_service_account_namespaces=default \
        policies=webapp \
        ttl=24h


vault write database/roles/postgres \
    db_name=default \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    revocation_statements="ALTER ROLE \"{{name}}\" NOLOGIN;"\
    default_ttl="1h" \
    max_ttl="24h"


vault write database/config/default \
    plugin_name=postgresql-database-plugin \
    allowed_roles="*" \
    connection_url="postgresql://{{username}}:{{password}}@postgresql.default.svc.cluster.local:5432/default?sslmode=disable" \
    username="postgres" \
    password="postgres"