from typing import Any

from pydantic_ai import Agent, RunContext

from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings

sre_agent = Agent(
    create_model(settings.resolve_model(settings.sre_model)),
    system_prompt="""You are a Senior SRE Agent tasked with the automated deployment and lifecycle management of the Orgitration ecosystem across multi-cloud Kubernetes clusters.
    You treat infrastructure as strictly **immutable code**, orchestrating modifications via robust GitOps workflows.

    Infrastructure Stack & Security Controls:
    - Tools & IaC: Utilize Terraform / OpenTofu combined with Crossplane for declarative, multi-region global state management.
    - Security Automation: Autonomously manage credential/secret rotation and enforce strict NetworkPolicies within Kubernetes boundaries.
    - Resilience & Scaling: Configure proactive multi-region failover mechanisms and define Horizontal Pod Autoscaler (HPA) targets driven by custom service metrics.

    ## Output Protocol
    Do NOT write files to disk. Return all generated code as structured output through this response. Your response will be captured as an Artifact and written to the filesystem by the pipeline orchestrator.
    """,
    output_type=Artifact,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@sre_agent.tool
def generate_helm_chart(ctx: RunContext[Any], service_name: str, image: str, port: int) -> dict[str, str]:
    """Returns a Helm chart (Chart.yaml, values.yaml and Deployment, Service, HPA, PDB, NetworkPolicy templates)."""
    base = f"helm/{service_name}"
    return {
        f"{base}/Chart.yaml": f'apiVersion: v2\nname: {service_name}\ndescription: {service_name} service\ntype: application\nversion: 0.1.0\nappVersion: "1.0.0"\n',
        f"{base}/values.yaml": f"""image:
  repository: {image}
  tag: ""
replicaCount: 2
service:
  port: {port}
resources:
  requests: {{cpu: 100m, memory: 128Mi}}
  limits: {{cpu: 500m, memory: 256Mi}}
autoscaling:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
""",
        f"{base}/templates/deployment.yaml": f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ .Release.Name }}}}
spec:
  replicas: {{{{ .Values.replicaCount }}}}
  selector:
    matchLabels: {{app: {{{{ .Release.Name }}}}}}
  template:
    metadata:
      labels: {{app: {{{{ .Release.Name }}}}}}
    spec:
      securityContext: {{runAsNonRoot: true}}
      containers:
        - name: {service_name}
          image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag | default .Chart.AppVersion }}}}"
          ports: [{{containerPort: {{{{ .Values.service.port }}}}}}]
          resources: {{{{- toYaml .Values.resources | nindent 12 }}}}
          livenessProbe: {{httpGet: {{path: /healthz, port: {{{{ .Values.service.port }}}}}}}}
          readinessProbe: {{httpGet: {{path: /readyz, port: {{{{ .Values.service.port }}}}}}}}
          securityContext: {{allowPrivilegeEscalation: false, readOnlyRootFilesystem: true}}
""",
        f"{base}/templates/service.yaml": """apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}
spec:
  selector: {app: {{ .Release.Name }}}
  ports: [{port: {{ .Values.service.port }}, targetPort: {{ .Values.service.port }}}]
""",
        f"{base}/templates/hpa.yaml": """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .Release.Name }}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: {{ .Release.Name }}}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: {type: Utilization, averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}}
""",
        f"{base}/templates/pdb.yaml": """apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ .Release.Name }}
spec:
  minAvailable: 1
  selector:
    matchLabels: {app: {{ .Release.Name }}}
""",
        f"{base}/templates/networkpolicy.yaml": """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ .Release.Name }}
spec:
  podSelector:
    matchLabels: {app: {{ .Release.Name }}}
  policyTypes: [Ingress]
  ingress:
    - ports: [{port: {{ .Values.service.port }}}]
""",
    }
