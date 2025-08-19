{{/*
Expand the name of the chart.
*/}}
{{- define "intelliflow-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "intelliflow-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "intelliflow-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "intelliflow-platform.labels" -}}
helm.sh/chart: {{ include "intelliflow-platform.chart" . }}
{{ include "intelliflow-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- if .Values.global.labels }}
{{ toYaml .Values.global.labels }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "intelliflow-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "intelliflow-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service selector labels for a specific service
*/}}
{{- define "intelliflow-platform.serviceSelectorLabels" -}}
app: {{ .serviceName }}
app.kubernetes.io/name: {{ .serviceName }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service labels for a specific service
*/}}
{{- define "intelliflow-platform.serviceLabels" -}}
{{ include "intelliflow-platform.labels" . }}
app: {{ .serviceName }}
component: {{ .component | default "microservice" }}
version: {{ .imageTag | default .Values.global.imageTag }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "intelliflow-platform.serviceAccountName" -}}
{{- if .Values.security.serviceAccounts.create }}
{{- default (include "intelliflow-platform.fullname" .) .Values.security.serviceAccounts.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccounts.name }}
{{- end }}
{{- end }}

{{/*
Service account name for a specific service
*/}}
{{- define "intelliflow-platform.serviceAccountNameForService" -}}
{{- if .Values.security.serviceAccounts.create }}
{{- .serviceName }}
{{- else }}
{{- default "default" .Values.security.serviceAccounts.name }}
{{- end }}
{{- end }}

{{/*
Generate image name for a service
*/}}
{{- define "intelliflow-platform.image" -}}
{{- $registry := .Values.global.imageRegistry }}
{{- $repository := .imageRepository }}
{{- $tag := .imageTag | default .Values.global.imageTag }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Generate environment variables for a service
*/}}
{{- define "intelliflow-platform.commonEnvVars" -}}
- name: SPRING_PROFILES_ACTIVE
  value: {{ .config.springProfiles | default "production" }}
- name: JAVA_OPTS
  value: {{ .config.javaOpts | default "-Xmx1g -Xms512m -XX:+UseG1GC" }}
- name: ENVIRONMENT
  value: {{ .Values.global.environment }}
- name: NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- end }}

{{/*
Generate database environment variables
*/}}
{{- define "intelliflow-platform.databaseEnvVars" -}}
- name: DB_USERNAME
  valueFrom:
    secretKeyRef:
      name: database-secrets
      key: DB_USERNAME
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: database-secrets
      key: DB_PASSWORD
- name: DATABASE_URL
  value: "jdbc:postgresql://{{ .Release.Name }}-postgresql:5432/intelliflow"
{{- end }}

{{/*
Generate Redis environment variables
*/}}
{{- define "intelliflow-platform.redisEnvVars" -}}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: redis-secrets
      key: REDIS_PASSWORD
- name: REDIS_URL
  value: "redis://:$(REDIS_PASSWORD)@{{ .Release.Name }}-redis-master:6379"
- name: REDIS_HOST
  value: "{{ .Release.Name }}-redis-master"
- name: REDIS_PORT
  value: "6379"
{{- end }}

{{/*
Generate Kafka environment variables
*/}}
{{- define "intelliflow-platform.kafkaEnvVars" -}}
- name: KAFKA_BOOTSTRAP_SERVERS
  value: "{{ .Release.Name }}-kafka:9092"
{{- end }}

{{/*
Generate JWT environment variables
*/}}
{{- define "intelliflow-platform.jwtEnvVars" -}}
- name: JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: jwt-secrets
      key: JWT_SECRET
{{- end }}

{{/*
Generate monitoring environment variables
*/}}
{{- define "intelliflow-platform.monitoringEnvVars" -}}
{{- if .Values.global.monitoring.enabled }}
- name: MANAGEMENT_ENDPOINTS_WEB_EXPOSURE_INCLUDE
  value: "health,info,metrics,prometheus"
- name: MANAGEMENT_ENDPOINT_METRICS_ENABLED
  value: "true"
- name: MANAGEMENT_ENDPOINT_PROMETHEUS_ENABLED
  value: "true"
- name: MANAGEMENT_METRICS_EXPORT_PROMETHEUS_ENABLED
  value: "true"
{{- end }}
{{- end }}

{{/*
Generate resource requirements
*/}}
{{- define "intelliflow-platform.resources" -}}
{{- if .resources }}
resources:
  {{- if .resources.requests }}
  requests:
    {{- if .resources.requests.memory }}
    memory: {{ .resources.requests.memory }}
    {{- end }}
    {{- if .resources.requests.cpu }}
    cpu: {{ .resources.requests.cpu }}
    {{- end }}
    {{- if .resources.requests.nvidia.com/gpu }}
    nvidia.com/gpu: {{ quote .resources.requests.nvidia.com/gpu }}
    {{- end }}
  {{- end }}
  {{- if .resources.limits }}
  limits:
    {{- if .resources.limits.memory }}
    memory: {{ .resources.limits.memory }}
    {{- end }}
    {{- if .resources.limits.cpu }}
    cpu: {{ .resources.limits.cpu }}
    {{- end }}
    {{- if .resources.limits.nvidia.com/gpu }}
    nvidia.com/gpu: {{ quote .resources.limits.nvidia.com/gpu }}
    {{- end }}
  {{- end }}
{{- else }}
resources:
{{ toYaml .Values.global.resources | indent 2 }}
{{- end }}
{{- end }}

{{/*
Generate security context
*/}}
{{- define "intelliflow-platform.securityContext" -}}
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: {{ .Values.global.securityContext.runAsUser }}
  runAsGroup: {{ .Values.global.securityContext.runAsGroup }}
  capabilities:
    drop:
    - ALL
{{- end }}

{{/*
Generate pod security context
*/}}
{{- define "intelliflow-platform.podSecurityContext" -}}
securityContext:
  runAsNonRoot: {{ .Values.global.securityContext.runAsNonRoot }}
  runAsUser: {{ .Values.global.securityContext.runAsUser }}
  runAsGroup: {{ .Values.global.securityContext.runAsGroup }}
  fsGroup: {{ .Values.global.securityContext.fsGroup }}
  seccompProfile:
    type: {{ .Values.global.securityContext.seccompProfile.type }}
{{- end }}

{{/*
Generate volume mounts for a service
*/}}
{{- define "intelliflow-platform.volumeMounts" -}}
- name: config-volume
  mountPath: /app/config
  readOnly: true
- name: logs-volume
  mountPath: /app/logs
- name: tmp-volume
  mountPath: /tmp
{{- if .extraVolumeMounts }}
{{ toYaml .extraVolumeMounts }}
{{- end }}
{{- end }}

{{/*
Generate volumes for a service
*/}}
{{- define "intelliflow-platform.volumes" -}}
- name: config-volume
  configMap:
    name: {{ .serviceName }}-config
- name: logs-volume
  emptyDir: {}
- name: tmp-volume
  emptyDir: {}
{{- if .extraVolumes }}
{{ toYaml .extraVolumes }}
{{- end }}
{{- end }}

{{/*
Generate affinity rules for a service
*/}}
{{- define "intelliflow-platform.affinity" -}}
{{- if .affinity }}
affinity:
{{ toYaml .affinity | indent 2 }}
{{- else }}
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - {{ .serviceName }}
        topologyKey: kubernetes.io/hostname
{{- end }}
{{- end }}

{{/*
Generate node selector for a service
*/}}
{{- define "intelliflow-platform.nodeSelector" -}}
{{- if .nodeSelector }}
nodeSelector:
{{ toYaml .nodeSelector | indent 2 }}
{{- else }}
nodeSelector:
  kubernetes.io/os: linux
{{- end }}
{{- end }}

{{/*
Generate tolerations for a service
*/}}
{{- define "intelliflow-platform.tolerations" -}}
{{- if .tolerations }}
tolerations:
{{ toYaml .tolerations | indent 0 }}
{{- else }}
tolerations:
- key: "intelliflow.com/dedicated"
  operator: "Equal"
  value: "microservices"
  effect: "NoSchedule"
{{- end }}
{{- end }}

{{/*
Generate service annotations
*/}}
{{- define "intelliflow-platform.serviceAnnotations" -}}
{{- if .Values.global.monitoring.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .servicePort | quote }}
prometheus.io/path: {{ .Values.global.monitoring.prometheus.path | default "/actuator/prometheus" }}
{{- end }}
{{- if .serviceAnnotations }}
{{ toYaml .serviceAnnotations }}
{{- end }}
{{- end }}

{{/*
Generate Istio sidecar annotations
*/}}
{{- define "intelliflow-platform.istioAnnotations" -}}
{{- if .Values.global.serviceMesh.enabled }}
sidecar.istio.io/inject: "true"
{{- if eq .Values.global.serviceMesh.mtls "STRICT" }}
security.istio.io/tlsMode: istio
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate namespace
*/}}
{{- define "intelliflow-platform.namespace" -}}
{{- .Values.global.namespace | default .Release.Namespace }}
{{- end }}

{{/*
Generate storage class
*/}}
{{- define "intelliflow-platform.storageClass" -}}
{{- .Values.global.storageClass | default "default" }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "intelliflow-platform.validateValues" -}}
{{- if not .Values.global.imageRegistry }}
{{- fail "global.imageRegistry is required" }}
{{- end }}
{{- if not .Values.global.imageTag }}
{{- fail "global.imageTag is required" }}
{{- end }}
{{- end }}
