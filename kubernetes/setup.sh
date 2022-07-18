#!/bin/bash

kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.8.2/cert-manager.crds.yaml
helm repo add jetstack https://charts.jetstack.
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo add cetic https://cetic.github.io/helm-charts
helm repo add twuni https://helm.twun.io
helm repo update

helm install cert-manager jetstack/cert-manager --version v1.8.2

kubectl apply -f cert-manager-ci.yaml
kubectl apply -f certs.yaml

helm install traefik traefik/traefik --values=traefik-values.yaml

kubectl apply -f secrets.yaml

kubectl apply -f local-pv.yaml
kubectl apply -f pv-claim.yaml

helm install private-registry twuni/docker-registry \
  --namespace default \
  --set replicaCount=1 \
  --set persistence.enabled=true \
  --set persistence.size=30Gi \
  --set persistence.deleteEnabled=true \
  --set persistence.storageClass=docker-registry-local-storage \
  --set persistence.existingClaim=docker-registry-pv-claim \
  --set secrets.htpasswd=$(cat ./htpasswd)

helm install private-registry twuni/docker-registry \
  --namespace default \
  --set replicaCount=1 \
  --set persistence.enabled=true \
  --set persistence.size=30Gi \
  --set persistence.deleteEnabled=true \
  --set secrets.htpasswd=$(cat ./htpasswd)

helm repo update
helm install consul hashicorp/consul --values helm-consul-values.yaml
helm install vault hashicorp/vault --values helm-vault-values.yaml

helm install pgadmin cetic/pgadmin

helm install postgresql bitnami/postgresql --values postgres-values.yaml
helm install redis bitnami/redis --set auth.enabled=false

kubectl apply -f pgadmin.yaml

kubectl apply -f traefik-routes.yaml
kubectl apply -f vault.yaml
kubectl apply -f registry.yaml

kubectl apply -f gisaxs-client.yaml
kubectl apply -f gisaxs-backend.yaml