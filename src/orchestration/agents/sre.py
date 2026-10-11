from pydantic_ai import Agent, RunContext

from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings

sre_agent = Agent(
    create_model(settings.resolve_model(settings.sre_model)),
    system_prompt="""You are a Senior SRE Agent tasked with the automated deployment and lifecycle management of the Orgitration ecosystem across multi-cloud Kubernetes clusters.
    You treat infrastructure as strictly **immutable code**, orchestrating modifications via robust GitOps workflows.

    Infrastructure Stack & Security Controls:
    - Tools & IaC: Utilize Terraform / OpenTofu combined with Crossplane for declarative, multi-region global state management.
    - Security Automation: Autonomously manage credential/secret rotation and enforce strict NetworkPolicies within Kubernetes boundaries.
    - Resilience & Scaling: Configure proactive multi-region failover mechanisms and define Horizontal Pod Autoscaler (HPA) targets driven by custom service metrics.

    ## Output Protocol
    Do NOT attempt to read or write files to disk. You do not have local I/O access. Return all generated code as structured output through this response. Your response will be captured as an Artifact and written to the filesystem by the pipeline orchestrator.
    """,
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


@sre_agent.tool
def provision_k8s_cluster(ctx: RunContext[str], provider: str, region: str, cluster_name: str) -> dict[str, str]:
    """Returns Terraform/OpenTofu HCL: VPC, EKS/GKE cluster, node groups, IAM roles, OIDC provider, IRSA."""
    return {
        "terraform/main.tf": f'# Terraform for {provider} in {region}\nresource "kubernetes_cluster" "main" {{}}',
        "terraform/vpc.tf": "# VPC + subnets + NAT",
        "terraform/iam.tf": "# IAM roles + OIDC + IRSA",
        "terraform/outputs.tf": "# Cluster endpoint, kubeconfig",
        "terraform/variables.tf": "# Region, cluster_name, instance types",
    }


@sre_agent.tool
def apply_helm_chart(ctx: RunContext[str], service_name: str, image: str, port: int) -> dict[str, str]:
    """Returns Helm chart: Chart.yaml, values.yaml, templates/ with Deployment, Service, Ingress, NetworkPolicy, HPA, ServiceMonitor, PDB."""
    return {
        f"helm/{service_name}/Chart.yaml": f"name: {service_name}\nversion: 1.0.0",
        f"helm/{service_name}/values.yaml": f"image: {image}\nport: {port}",
        f"helm/{service_name}/templates/deployment.yaml": "# Deployment",
        f"helm/{service_name}/templates/service.yaml": "# Service",
        f"helm/{service_name}/templates/ingress.yaml": "# Ingress",
        f"helm/{service_name}/templates/hpa.yaml": "# HPA",
        f"helm/{service_name}/templates/servicemonitor.yaml": "# ServiceMonitor",
        f"helm/{service_name}/templates/networkpolicy.yaml": "# NetworkPolicy",
        f"helm/{service_name}/templates/pdb.yaml": "# PDB",
    }
